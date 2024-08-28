import functools
import logging
import os

import subprocess

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

from  DAMESRL.liir.dame.core.io.CoNLL2005Reader import CoNLL2005Reader
from  DAMESRL.liir.dame.core.io.CoNLL2005Writer import write_props, write_short_conll2005_format
from  DAMESRL.liir.dame.core.nn.Data import BucketedDataIteratorSEQ, PaddedDataIteratorSEQ
from  DAMESRL.liir.dame.core.nn.ExtHighWayWrapper import ExtHighWayLSTMCell, ExtDropoutWrapper
from  DAMESRL.liir.dame.core.nn.thirdparties.Attention import add_timing_signal, multihead_attention, attention_bias
from  DAMESRL.liir.dame.core.nn.thirdparties.Common import linear, layer_norm
from  DAMESRL.liir.dame.srl.CharWordEmbedding import CharProcessor
from  DAMESRL.liir.dame.srl.ConfigReader import Configuration
from  DAMESRL.liir.dame.srl.DataProcessor import DataProcessor


def write_score(scs):
    return  str(scs["P"]) + " " + str(scs["R"]) + " " + str(scs["conll"])

def lazy_property(function):
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return wrapper


class MyExpSummary:
    def __init__(self):
        self.epoches = []
        self.losses = []
        self.evals = []
        self.conllWSJ = []
        self.conllBrown = []
        self.best_global_step = 0
        self.current_epoch = 0
        self.best_eval = -0.1
        self.lr = 1.0

    def export_data(self, output):
        f = open(os.path.join(output, "summary.txt"), "w")

        f.write(",".join([str(x) for x in self.epoches]))
        f.write("\n")
        f.write(",".join([str(x) for x in self.losses]))
        f.write("\n")
        f.write(" ".join([str(x) for x in self.evals]))
        f.write("\n")
        f.write(" ".join([str(x) for x in self.conllWSJ]))
        f.write("\n")
        f.write(" ".join([str(x) for x in self.conllBrown]))
        f.write("\n")
        f.write(str(self.best_global_step))
        f.write("\n")
        f.write(str(self.current_epoch))
        f.write("\n")
        f.write(str(self.best_eval))
        f.write("\n")
        f.write(str(self.lr))

        f.close()

    def import_data(self, output):
        def extract_score(val):
            return [float(x) for x in val.split(" ")]

        try:
            f = open(os.path.join(output, "summary.txt"), "r")
            lines = [l.strip() for l in f.readlines()]
            try:
                self.epoches = [int(x) for x in lines[0].split(",")]
            except Exception:
                pass
            try:
                self.losses = [float(x) for x in lines[1].split(",")]
            except Exception:
                pass
            try:
                self.evals = [ extract_score(x) for x in lines[2].split(",")]
            except Exception:
                pass
            try:
                self.conllWSJ = [ extract_score(x) for x in lines[3].split(",")]
            except Exception:
                pass
            try:
                self.conllBrown = [ extract_score(x) for x in lines[4].split(",")]
            except Exception:
                pass

            self.best_global_step = int (lines[5])
            self.current_epoch = int (lines[6])
            self.best_eval = float(lines[7])
            self.lr = float(lines[8])
            f.close()
        except Exception:
            pass





class DSRL:
    def __init__(self, config_path="config/srl.ini", tag_id=None):
        '''
        init a deep semantic role labeller
        :param config_path: path to configuaration file
        '''
        self.word_features = tf.placeholder(dtype=tf.int32, shape=[None, None],
                                            name="words")  # word features - embeddings
        self.pred_features = tf.placeholder(dtype=tf.int32, shape=[None, None],
                                            name="preds")  # boolean features indicating whether a word is a predicate or not
        self.seq_lens = tf.placeholder(dtype=tf.int32, shape=[None], name="seq_lens")  # sentence lengths
        self.targets = tf.placeholder(dtype=tf.int32, shape=[None, None], name='targets')  # target labels
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[],
                                      name="dropout")  # dropout value, is 1.0 for evaluation
        self.is_train = tf.placeholder(dtype=tf.bool, shape=[],
                                       name="istrain")  # learning rate
        self.cfg = Configuration(config_path)  # configuration
        self.lr = tf.placeholder(dtype=tf.float32, shape=[],
                                 name="lr")  # learning rate
        if tag_id is not None:
            self.cfg.tag = tag_id


        if "char" in self.cfg.emb_type:
            # for character embeddings

            self.char_features = tf.placeholder(tf.int32, shape=[None, None, None],
                                                name="chars")
            self.char_lens = tf.placeholder(tf.int32, shape=[None, None],
                                            name="char_lens")



        if self.cfg.top_type == "crf":
            self.trans_params = None


        self.train_data = None
        self.dev_data = None
        self.dev_sentences = None
        self.logger = None
        self.sess = None
        self.saver = None
        self.my_summary = MyExpSummary()
        self.we_dict = None
        self.init()

        self.embeddings
        self.logits
        self.prediction
        self.loss
        self.error
        self.train_op

        if self.cfg.mode == "infer":
            self.restore()

    #this function is only used for thirdparties
    def deepatt_default_params(self, is_train):

        params = tf.contrib.training.HParams(

            hidden_size=self.cfg.hidden_dim,
            filter_size=self.cfg.filter_size,
            filter_width=self.cfg.filter_width,
            num_heads=self.cfg.num_heads,
            num_hidden_layers=self.cfg.num_layers,
            attention_dropout=self.cfg.attention_dropout,
            residual_dropout=self.cfg.residual_dropout,
            relu_dropout=self.cfg.relu_dropout,
            label_smoothing=self.cfg.label_smoothing,
            fix_embedding=self.cfg.train_word_embeddings,
            layer_preprocessor=self.cfg.layer_preprocessor,
            layer_postprocessor=self.cfg.layer_postprocessor,# "layer_norm",
            attention_key_channels=None,
            attention_value_channels=None,
            attention_function=self.cfg.attention_function,# "dot_product",
            layer_type=self.cfg.layer_type, #"ffn_layer",
            multiply_embedding_mode=self.cfg.multiply_embedding_mode,#"sqrt_depth",

        )


        params.attention_dropout = tf.cond(is_train, lambda: self.cfg.attention_dropout, lambda: 0.0)
        params.residual_dropout = tf.cond(is_train, lambda: self.cfg.residual_dropout, lambda: 0.0)
        params.relu_dropout = tf.cond(is_train, lambda: self.cfg.relu_dropout, lambda: 0.0)
        return params


    @lazy_property
    def embeddings(self):
        with tf.variable_scope("words"):
            _word_embeddings1 = None
            if self.cfg.word_embedding_path is None:
                _word_embeddings = tf.get_variable(
                    name="_word_embeddings",
                    dtype=tf.float32,
                    shape=[len(self.pp.words), self.cfg.embedding_size],
                    trainable=self.cfg.train_word_embeddings,
                    initializer=tf.random_normal_initializer(0.0, self.cfg.embedding_size ** -0.5))

                if len(self.pp.extra_words)>0:
                    _word_embeddings_extra = tf.get_variable(
                        name="_word_embeddings_extra",
                        dtype=tf.float32,
                        shape=[len(self.pp.extra_words), self.cfg.embedding_size],
                        trainable=self.cfg.train_word_embeddings,
                        initializer=tf.random_normal_initializer(0.0, self.cfg.embedding_size ** -0.5))

                    _word_embeddings1 = tf.concat([_word_embeddings, _word_embeddings_extra], 0)
                else:
                    _word_embeddings1 = _word_embeddings
            else:
                _word_embeddings = tf.Variable(
                    self.pp.get_we_dict(self.cfg.word_embedding_path),
                    name="_word_embeddings",
                    dtype=tf.float32,
                    trainable=self.cfg.train_word_embeddings)
                _word_embeddings1 = _word_embeddings
                if len(self.pp.extra_words)>0:
                    _word_embeddings_extra = tf.Variable(
                    self.pp.get_we_dict_extra(self.cfg.word_embedding_path),
                    name="_word_embeddings_extra",
                    dtype=tf.float32,
                    trainable=self.cfg.train_word_embeddings)
                    _word_embeddings1 = tf.concat([_word_embeddings, _word_embeddings_extra],0)


            word_embeddings = tf.gather(_word_embeddings1, self.word_features)
        with tf.variable_scope("preds"):

            _pred_embeddings = tf.get_variable(
                name="_pred_embeddings",
                dtype=tf.float32,
                shape=[2, self.cfg.embedding_pred_size],
                trainable=self.cfg.train_pred_embeddings,
                initializer=tf.random_normal_initializer(0.0, self.cfg.embedding_pred_size ** -0.5))
            pred_embeddings = tf.gather(_pred_embeddings, self.pred_features)

        if "char" in self.cfg.emb_type:

            with tf.variable_scope("chars"):
                # get char embeddings matrix
                _char_embeddings = tf.get_variable(
                    name="_char_embeddings",
                    dtype=tf.float32,
                    shape=[len(self.charpp.vob), self.cfg.embedding_char_size],
                    initializer=tf.random_normal_initializer(0.0, 0.01)
                )
                char_embeddings = tf.nn.embedding_lookup(_char_embeddings,
                                                         self.char_features, name="char_embeddings")

                # put the time dimension on axis=1
                s = tf.shape(char_embeddings)
                char_embeddings = tf.reshape(char_embeddings,
                                             shape=[s[0] * s[1], s[-2], self.cfg.hidden_char_dim])
                word_lengths = tf.reshape(self.char_lens, shape=[s[0] * s[1]])

                # bi lstm on chars
                cell_fw = tf.contrib.rnn.LSTMCell(self.cfg.hidden_char_dim,
                                                  state_is_tuple=True, initializer=tf.orthogonal_initializer)
                cell_bw = tf.contrib.rnn.LSTMCell(self.cfg.hidden_char_dim,
                                                  state_is_tuple=True, initializer=tf.orthogonal_initializer)
                _output = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, char_embeddings,
                    sequence_length=word_lengths, dtype=tf.float32)

                # read and concat output
                _, ((_, output_fw), (_, output_bw)) = _output
                output = tf.concat([output_fw, output_bw], axis=-1)

                # shape = (batch size, max sentence length, char hidden size)
                output = tf.reshape(output,
                                    shape=[s[0], s[1], 2 * self.cfg.hidden_char_dim])
                word_char_embeddings =  output

            return tf.concat([word_embeddings, word_char_embeddings, pred_embeddings], -1)

        return tf.concat([word_embeddings, pred_embeddings], -1)


    def logits_for_rnn(self, input=None):

        def create_lstm_cell():
            cell_lstm = rnn.LSTMCell(self.cfg.hidden_dim, initializer=tf.orthogonal_initializer)

            cell_lstm = ExtDropoutWrapper(cell_lstm, output_keep_prob=self.dropout, \
                                          variational_recurrent=True, dtype=tf.float32, is_train=self.is_train,
                                          seed=12345)

            return cell_lstm

        def create_lstm_first_cell():
            cell_lstm = rnn.LSTMCell(self.cfg.hidden_dim, initializer=tf.orthogonal_initializer)
            cell_lstm = ExtDropoutWrapper(cell_lstm, output_keep_prob=self.dropout, input_keep_prob=self.dropout,\
                                          variational_recurrent=True, dtype=tf.float32, is_train=self.is_train,
                                          seed=12345, input_size=self.embeddings.get_shape()[-1])

            return cell_lstm


        def create_highway_cell():
            cell_lstm = ExtHighWayLSTMCell(self.cfg.hidden_dim, initializer=tf.orthogonal_initializer)
            cell_lstm = ExtDropoutWrapper(cell_lstm, output_keep_prob=self.dropout, \
                                          variational_recurrent=True, dtype=tf.float32, is_train=self.is_train,
                                          seed=12345)

            return cell_lstm

        def create_highway_first_cell():
            cell_lstm = ExtHighWayLSTMCell(self.cfg.hidden_dim, initializer=tf.orthogonal_initializer)
            cell_lstm = ExtDropoutWrapper(cell_lstm, output_keep_prob=self.dropout, input_keep_prob=self.dropout,\
                                          variational_recurrent=True, dtype=tf.float32, is_train=self.is_train,
                                          seed=12345, input_size=self.embeddings.get_shape()[-1])

            return cell_lstm

        if input is None:
            input = self.embeddings

        all_inputs = input

        for i in range(self.cfg.num_layers):
            with tf.variable_scope("bi-di-lstm-" + str(i)):

                if self.cfg.layer_type == "lstm":
                    if i == 0:
                        cell = create_lstm_first_cell()
                    else:
                        cell = create_lstm_cell()

                else:
                    if self.cfg.layer_type == "highway":
                        if i == 0:
                            cell = create_highway_first_cell()
                        else:
                            cell = create_highway_cell()

                ioutputs, _ = tf.nn.dynamic_rnn(cell,
                                                all_inputs[i], self.seq_lens, dtype=tf.float32)

                ioutputs = tf.reverse_sequence(ioutputs, self.seq_lens, seq_dim=1,
                                               batch_dim=0)

                all_inputs.append(ioutputs)

        return all_inputs[-1]


    def logits_for_att(self, input=None):
        params = self.deepatt_default_params(self.is_train)

        def _residual_fn(x, y, params):
            y = tf.nn.dropout(y, 1.0 - params.residual_dropout)
            return layer_norm(x + y)

        def _ffn_layer(inputs, hidden_size, output_size, keep_prob=None,
                       data_format="NHWC", dtype=None, scope=None):
            with tf.variable_scope(scope, default_name="ffn_layer", values=[inputs],
                                   dtype=dtype):
                with tf.variable_scope("input_layer"):
                    hidden = linear(inputs, hidden_size, True, data_format=data_format)
                    hidden = tf.nn.relu(hidden)
                    hidden = tf.cond(keep_prob < 1.0, lambda: tf.nn.dropout(hidden, keep_prob), lambda: hidden)

                with tf.variable_scope("output_layer"):
                    output = linear(hidden, output_size, True, data_format=data_format)

                return output

        def encoder(encoder_input, mask, params, dtype=None, scope=None):
            with tf.variable_scope(scope, default_name="encoder", dtype=dtype,
                                   values=[encoder_input, mask]):
                x = encoder_input
                attn_bias = attention_bias(mask, "masking")

                for layer in range(params.num_hidden_layers):
                    with tf.variable_scope("layer_%d" % layer):
                        with tf.variable_scope("computation"):
                            y = _ffn_layer(
                                x,
                                params.filter_size,
                                params.hidden_size,
                                1.0 - params.relu_dropout,
                            )
                            x = _residual_fn(x, y, params)


                            # Do not use non-linear layer otherwise

                        with tf.variable_scope("self_attention"):
                            y = multihead_attention(
                                x,
                                None,
                                attn_bias,
                                params.attention_key_channels or params.hidden_size,
                                params.attention_value_channels or params.hidden_size,
                                params.hidden_size,
                                params.num_heads,
                                1.0 - params.attention_dropout,
                                attention_function=params.attention_function
                            )
                            x = _residual_fn(x, y, params)

                return x
        if input is None:
            input = self.embeddings

        mask = tf.to_float(tf.not_equal(self.word_features, 0))

        inputs = input
        if params.multiply_embedding_mode == "sqrt_depth":
            inputs = inputs * (self.cfg.hidden_dim ** 0.5)

        inputs = inputs * tf.expand_dims(mask, -1)

        bias = tf.get_variable("bias", [self.cfg.hidden_dim])
        encoder_input = tf.nn.bias_add(inputs, bias)

        encoder_input = add_timing_signal(encoder_input)

        keep_prob = 1.0 - params.residual_dropout
        encoder_input = tf.nn.dropout(encoder_input, keep_prob)

        encoder_output = encoder(encoder_input, mask, params)

        return encoder_output

    @lazy_property
    def logits(self):


        if self.cfg.model_type == "rnn":
            outputs = self.logits_for_rnn([self.embeddings])
        elif self.cfg.model_type == "att":
            outputs = self.logits_for_att(self.embeddings)


        with tf.variable_scope("proj"):
            W = tf.get_variable("W", dtype=tf.float32,
                                shape=[self.cfg.hidden_dim, self.pp.get_num_labels()],
                                initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=12345))

            b = tf.get_variable("b", shape=[self.pp.get_num_labels()],
                                dtype=tf.float32, initializer=tf.zeros_initializer)

            nsteps = tf.shape(outputs)[1]
            output = tf.reshape(outputs, [-1, self.cfg.hidden_dim])
            pred = tf.matmul(output, W) + b
            return tf.reshape(pred, [-1, nsteps, self.pp.get_num_labels()])



    @lazy_property
    def prediction(self):
        return tf.cast(tf.argmax(self.logits, axis=-1),
                       tf.int32)

    @lazy_property
    def loss(self):
        def loss_crf():
            log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
                self.logits, self.targets, self.seq_lens)
            self.trans_params = trans_params  # need to evaluate it for decoding
            loss = tf.reduce_mean(-log_likelihood)
            tf.summary.scalar("loss", loss)
            return loss

        if self.cfg.top_type == "crf":
            return loss_crf()

        if self.cfg.label_smoothing != 0.0:
            label_smoothing = self.cfg.label_smoothing
            n = tf.to_float(len(self.pp.labels) - 1)
            p = 1.0 - label_smoothing
            q = label_smoothing / n
            # Soft targets.
            soft_targets = tf.one_hot(self.targets, depth=len(self.pp.labels), axis=-1,
                                      on_value=p, off_value=q)

            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,
                                                             labels=soft_targets)
        else:
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self.targets)

        mask = tf.sequence_mask(self.seq_lens)
        losses = tf.boolean_mask(losses, mask)
        loss = tf.reduce_mean(losses)
        tf.summary.scalar("loss", loss)
        return loss

    @lazy_property
    def train_op(self):
        global_step = tf.train.get_or_create_global_step()

        if self.cfg.optimizer == "adadelta":
            optimizerf = tf.train.AdadeltaOptimizer
            optimizer = optimizerf(learning_rate=self.lr, epsilon=self.cfg.epsilon)
        elif self.cfg.optimizer == "adagrad":
            optimizerf = tf.train.AdagradOptimizer
            optimizer = optimizerf(learning_rate=self.lr)
        elif self.cfg.optimizer == "adam":
            optimizerf = tf.train.AdamOptimizer
            optimizer = optimizerf(learning_rate=self.lr)
        else:
            optimizerf = tf.train.GradientDescentOptimizer
            optimizer = optimizerf(learning_rate=self.lr)

        grads, vs = zip(*optimizer.compute_gradients(self.loss))
        grads, norm = tf.clip_by_global_norm(grads, 1.0)
        return optimizer.apply_gradients(zip(grads, vs), global_step=global_step)

    def init(self):
        def get_data_list():
            if self.cfg.mode == "train":

                lst= []
                if self.cfg.train_path is not None:
                    lst.append(self.cfg.train_path)

                if self.cfg.dev_path is not None:
                    lst.append(self.cfg.dev_path)

                if self.cfg.test_brown_path is not None:
                    lst.append(self.cfg.test_brown_path)


                if self.cfg.test_wsj_path is not None:
                    lst.append(self.cfg.test_wsj_path)
                return lst
            if self.cfg.mode == "infer":
                if self.cfg.test_path is not None:

                    return self.cfg.test_path

        if not os.path.exists(self.cfg.model_dir):
            if self.cfg.mode == "infer":
                self.logger.error("Model dir does not exist!")
                sys.exit(0)
            else:
                os.makedirs(self.cfg.model_dir)

        self.pp = DataProcessor(model_dir=self.cfg.model_dir)
        self.pp.max_len = self.cfg.max_train_length



        if self.cfg.mode == "train":


            if self.pp.is_empty():
                self.pp.generate_vob(get_data_list())
                if self.cfg.smooth_rare_words > 0:
                    self.pp.smooth_word_vob(self.cfg.smooth_rare_words)
                self.pp.export_vobs(self.cfg.model_dir)

            else:
                if self.cfg.extend_vob:
                    self.pp.extend_vob(get_data_list())
                    self.pp.export_vobs(self.cfg.model_dir)



            self.train_data = self.pp.get_data_train(self.cfg.train_path, data=self.train_data)

            self.dev_sentences = CoNLL2005Reader(self.cfg.dev_path).read_all()

            self.dev_data = self.pp.get_data_eval(self.cfg.dev_path, self.dev_sentences)


            self.brown_sentences = CoNLL2005Reader(self.cfg.test_brown_path).read_all() if self.cfg.test_brown_path is not None else None
            self.wsj_sentences = CoNLL2005Reader(self.cfg.test_wsj_path).read_all() if self.cfg.test_wsj_path is not None else None
            self.brown_data  = self.pp.get_data_eval(self.cfg.test_brown_path, self.brown_sentences) if self.cfg.test_brown_path is not None else None
            self.wsj_data = self.pp.get_data_eval(self.cfg.test_wsj_path, self.wsj_sentences) if self.cfg.test_wsj_path is not None else None

        if self.cfg.mode == "infer" and self.cfg.extend_vob:
            self.pp.extend_vob(get_data_list())
            self.pp.export_vobs(self.cfg.model_dir)

        if "char" in self.cfg.emb_type:
            self.charpp = CharProcessor(self.pp.words)

        self.my_summary.import_data(self.cfg.model_dir)


        # start logging
        self.logger = logging.getLogger('logger')
        self.logger.setLevel(logging.DEBUG)
        logging.basicConfig(format='%(message)s', level=logging.DEBUG)
        handler = logging.FileHandler(self.cfg.log_file)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s:%(levelname)s: %(message)s'))
        logging.getLogger().addHandler(handler)


    def save_session(self, tag):
        saver = tf.train.Saver(max_to_keep=self.cfg.max_to_save)
        saver.save(self.sess, os.path.join(self.cfg.model_dir, "srl_" + tag))

    def close(self):
        self.sess.close()



    @lazy_property
    def error(self):
        mistakes = tf.not_equal(
            tf.argmax(self.targets, -1), tf.argmax(self.prediction, -1))

        return tf.reduce_mean(tf.cast(mistakes, tf.float32))

    def restore(self):

        def optimistic_restore_vars(model_checkpoint_path):
            reader = tf.train.NewCheckpointReader(model_checkpoint_path)
            saved_shapes = reader.get_variable_to_shape_map()
            var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
                                if var.name.split(':')[0] in saved_shapes])
            restore_vars = []
            name2var = dict(zip(map(lambda x: x.name.split(':')[0], tf.global_variables()), tf.global_variables()))
            with tf.variable_scope('', reuse=True):
                for var_name, saved_var_name in var_names:
                    curr_var = name2var[saved_var_name]
                    var_shape = curr_var.get_shape().as_list()
                    if var_shape == saved_shapes[saved_var_name]:
                        restore_vars.append(curr_var)
            return restore_vars

        #ckpt = tf.train.get_checkpoint_state(os.path.dirname(checkpoint + '/checkpoint'))
        #saver = tf.train.Saver(max_to_keep=20,
        #                       var_list=optimistic_restore_vars(ckpt.model_checkpoint_path) if checkpoint else None)

        #saver = tf.train.Saver(max_to_keep=self.cfg.max_to_save, var_list=optimistic_restore_vars())
        self.logger.info("Reloading the {} trained model...".format(self.cfg.tag))
        self.sess = tf.Session()
        if self.cfg.tag != "last":
            if self.cfg.tag == "best":
                saver = tf.train.Saver(max_to_keep=self.cfg.max_to_save, var_list=optimistic_restore_vars(os.path.join(self.cfg.model_dir, "best")))

                saver.restore(self.sess, os.path.join(self.cfg.model_dir, "best"))

            else:
                saver = tf.train.Saver(max_to_keep=self.cfg.max_to_save,
                                       var_list=optimistic_restore_vars(os.path.join(self.cfg.model_dir, self.cfg.tag)))

                saver.restore(self.sess, os.path.join(self.cfg.model_dir, self.cfg.tag))

        else:

            lst = tf.train.latest_checkpoint(self.cfg.model_dir)

            ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.cfg.model_dir + '/checkpoint'))
            saver = tf.train.Saver(max_to_keep=self.cfg.max_to_save,
                                   var_list=optimistic_restore_vars(ckpt.model_checkpoint_path) )

            #saver = tf.train.Saver(max_to_keep=self.cfg.max_to_save,
            #                       var_list=optimistic_restore_vars(os.path.join(self.cfg.model_dir, "best")))

            if lst is not None:
                saver.restore(self.sess, lst)
            else:
                self.logger.error("Model tag doesn't exist!")

    def train(self):
        def optimistic_restore_vars(model_checkpoint_path):
            reader = tf.train.NewCheckpointReader(model_checkpoint_path)
            saved_shapes = reader.get_variable_to_shape_map()
            var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
                                if var.name.split(':')[0] in saved_shapes])
            restore_vars = []
            name2var = dict(zip(map(lambda x: x.name.split(':')[0], tf.global_variables()), tf.global_variables()))
            with tf.variable_scope('', reuse=True):
                for var_name, saved_var_name in var_names:
                    curr_var = name2var[saved_var_name]
                    var_shape = curr_var.get_shape().as_list()
                    if var_shape == saved_shapes[saved_var_name]:
                        restore_vars.append(curr_var)
            return restore_vars
        lst = tf.train.latest_checkpoint(self.cfg.model_dir)

        ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.cfg.model_dir + '/checkpoint'))
        saver = tf.train.Saver(max_to_keep=self.cfg.max_to_save,
                               var_list=optimistic_restore_vars(ckpt.model_checkpoint_path) if ckpt is not None else None)

        with tf.Session() as self.sess:

            tvars = tf.trainable_variables()
            self.sess.run(tf.initialize_all_variables())

            for var in tvars:
                print(var.name)  # Prints the name of the variable alongside its value.


            if self.cfg.tag != "last":
                #if os._exists(os.path.join(self.cfg.model_dir, self.cfg.tag)):
                saver.restore(self.sess, os.path.join(self.cfg.model_dir, self.cfg.tag))
                #else:
                #    self.sess.run(tf.global_variables_initializer())
            else:
                lst = tf.train.latest_checkpoint(self.cfg.model_dir)

                if lst is not None:
                    saver.restore(self.sess, lst)
                else:
                    self.sess.run(tf.global_variables_initializer())

            best_eval = self.my_summary.best_eval
            nepoch_no_imprv = 0  # for early stopping


            train_input = BucketedDataIteratorSEQ(dt=self.train_data, num_buckets=self.cfg.num_buckets)
            dev_input = PaddedDataIteratorSEQ(self.dev_data)
            brown_input = PaddedDataIteratorSEQ(self.brown_data) if self.cfg.test_brown_path is not None else None
            wsj_input = PaddedDataIteratorSEQ(self.wsj_data) if self.cfg.test_wsj_path is not None else None
            for epoch in range(self.my_summary.current_epoch, self.cfg.nb_epochs):

                self.logger.info("Epoch {:} out of {:}".format(epoch ,
                                                               self.cfg.nb_epochs))

                loss, eval, eval_brown, eval_wsj = self.run_epoch(train_input, dev_input, brown_input, wsj_input, epc=epoch)

                self.my_summary.epoches.append(str(epoch + 1))
                self.my_summary.losses.append(str(loss))
                self.my_summary.evals.append( write_score(eval))
                self.my_summary.conllBrown.append(write_score(eval_brown))
                self.my_summary.conllWSJ.append(write_score(eval_wsj))


                if epoch % self.cfg.save_freq == 0:
                    self.my_summary.current_epoch = epoch
                    saver.save(self.sess, os.path.join(self.cfg.model_dir, "srl"), global_step=epoch)
                # early stopping and saving best parameters
                eval = eval["conll"]
                if eval > best_eval:
                    nepoch_no_imprv = 0

                    saver.save(self.sess, os.path.join(self.cfg.model_dir, "best"))
                    self.my_summary.best_eval = eval
                    self.my_summary.best_global_step = epoch
                    self.my_summary.export_data(self.cfg.model_dir)

                    best_eval = eval
                    self.logger.info("- new best score!")
                else:
                    nepoch_no_imprv += 1
                    if nepoch_no_imprv >= self.cfg.patience:
                        self.logger.info("- early stopping {} epochs without " \
                                         "improvement".format(nepoch_no_imprv))
                        break
                self.my_summary.export_data(self.cfg.model_dir)

        self.logger.info("Epoch indexes: {}".format(" ".join([str(x) for x in self.my_summary.epoches])))
        self.logger.info("Losses: {}".format(" ".join([str(x) for x in self.my_summary.losses])))
        self.logger.info("Evals: {}".format(" ".join([str(x) for x in self.my_summary.evals])))

    def get_feed(self, inputs_vals, predicate_vals, target_vals, seq_len_vals, dr=1.0, char_vals=None, char_lens = None, is_train=True, lr=None):
        if lr is None:
            lr = self.cfg.learning_rate
        if char_vals is not None:
            return {
                self.word_features: inputs_vals,
                self.pred_features: predicate_vals,
                self.char_features: char_vals,
                self.char_lens: char_lens,
                self.seq_lens: seq_len_vals,
                self.targets: target_vals,
                self.dropout: dr,
                self.is_train: is_train,
                self.lr : lr

            }
        else:
            return {
                self.word_features: inputs_vals,
                self.pred_features: predicate_vals,
                self.seq_lens: seq_len_vals,

                self.targets: target_vals,
                self.dropout: dr,
                self.is_train: is_train,
                self.lr: lr

            }

    def get_learning_rate(self, epc):
        if epc >= self.cfg.learning_rate_decay_start and epc % self.cfg.learning_rate_decay_batch_step == 0 \
                and self.my_summary.lr is not None:
            self.my_summary.lr /= self.cfg.learning_rate_decay_step
            lr = self.my_summary.lr
        else:
            if self.my_summary.lr is not None:
                lr = self.my_summary.lr
            else:
                lr = self.cfg.learning_rate
        return lr

    def run_epoch(self, train_input, dev_input, brown_input, wsj_input, epc=None):
        # iterate over dataset
        current_epoch = train_input.epochs

        self.logger.info("training epoch {}".format(train_input.epochs))
        all_losses = []
        lr = self.get_learning_rate(epc)
        print (lr)
        while train_input.epochs == current_epoch:

            (inputs_vals, predicate_vals, target_vals), seq_len_vals = train_input.next_batch(self.cfg.batch_size)

            char_vals, char_lens = None, None
            if "char" in self.cfg.emb_type:
                char_vals, char_lens = self.charpp.get_data(inputs_vals)


            feeddct = self.get_feed(inputs_vals, predicate_vals, target_vals, seq_len_vals,
                                    self.cfg.dropput_keep_prob, char_vals, char_lens, lr=lr)


            _, train_loss= self.sess.run(
                [self.train_op, self.loss], feed_dict=feeddct)
            all_losses.append(train_loss)
        metrics = self.conll_evaluate(dev_input, self.dev_sentences , self.cfg.dev_path)
        metrics_brown = self.conll_evaluate(brown_input, self.brown_sentences, self.cfg.test_brown_path) if self.cfg.test_brown_path is not None else {"conll":0.0, "P":0.0, "R":0.0}
        metrics_wsj = self.conll_evaluate(wsj_input, self.wsj_sentences, self.cfg.test_wsj_path) if self.cfg.test_brown_path is not None else {"conll":0.0, "P":0.0, "R":0.0}
        avg_loss = np.mean(np.asarray(all_losses))

        msg = "avg loss {:06.7f}, eval acc: {:04.4f}, Precise:{:04.4f}, Recall: {:04.4f}".format(avg_loss, metrics["conll"], metrics["P"], metrics["R"])
        self.logger.info(msg)

        return avg_loss, metrics, metrics_brown, metrics_wsj

    def conll_evaluate(self, test_input, test, test_path):

        if not os.path.exists(test_path + "_props.txt"):
            gtest = CoNLL2005Reader(test_path).read_all()
            write_props(gtest,test_path + "_props.txt" )

        current_epoch = test_input.epochs

        predictions = []

        all_predictions = []

        while test_input.epochs == current_epoch:
            (inputs_vals, predicate_vals, target_vals), seq_len_vals = test_input.next_batch(self.cfg.batch_dev_size)

            char_vals, char_lens = None, None
            if "char" in self.cfg.emb_type:
                char_vals, char_lens = self.charpp.get_data(inputs_vals)

            feeddct_err = self.get_feed(inputs_vals, predicate_vals, target_vals, seq_len_vals,
                                        self.cfg.learning_rate, char_vals, char_lens, is_train=False)


            if self.cfg.top_type == "crf":
                scoresp, trans_params = self.sess.run([self.logits, self.trans_params], feed_dict=feeddct_err)

                tmp = []

                for score, sequence_length in zip(scoresp.tolist(), seq_len_vals):
                    score = np.asarray(score[:sequence_length])  # keep only the valid steps
                    viterbi_seq, viterbi_score = tf.contrib.crf.viterbi_decode(
                        score, trans_params)
                    tmp.append(viterbi_seq)

                all_predictions.extend(tmp)
            else:
                if self.cfg.inference_type == "A*":
                    [score] = self.sess.run(
                            [self.logits], feed_dict=feeddct_err)

                    predictions.extend(score.tolist())
                elif self.cfg.inference_type == "argmax":
                    score, predicted_labels = self.sess.run(
                        [self.logits, self.prediction], feed_dict=feeddct_err)
                    predicted_labels = predicted_labels.tolist()
                    x = []
                    for lbls in predicted_labels:
                        x.append([self.pp.labels[idx] for idx in lbls])
                    all_predictions.extend(x)

        if self.cfg.top_type == "softmax" and self.cfg.inference_type == "A*":
            for i, scores in enumerate(predictions):
                predictions_infe = self.inference(scores, self.get_transition_params(self.pp.labels))

                all_predictions.append([self.pp.labels[idx] for idx in predictions_infe[0]])

        pos = 0
        for sen in test:
            for pred in sen.get_predicates():
                pred.arguments = []
        for sen in test:
            for pred in sen.get_predicates():
                temps = all_predictions[pos][0:len(sen)]

                if self.cfg.top_type == "crf":
                    temps = [self.pp.labels[idx] for idx in temps]
                pred.arguments = self.post_processing(temps)
                pos += 1

        write_props(test, test_path + "_out.txt")
        write_short_conll2005_format(test, test_path + "_full_output.txt")
        finalscores = self.get_CoNLL2001Score(test_path + "_props.txt", test_path + "_out.txt")
        return {"conll": finalscores[2], "P": finalscores[0], "R":finalscores[1] }



    def get_CoNLL2001Score(self, goldp, sysp):
        child = subprocess.Popen('sh {} {} {}'.format(self.cfg.eval_script, goldp, sysp),
                                 shell=True, stdout=subprocess.PIPE)
        eval_info = child.communicate()[0].decode("utf-8")
        try:
            Fscore = eval_info.strip().split("\n")[6]
            temps = Fscore.strip().split()

            return float(temps[4]), float(temps[5]), float(temps[6])
        except IndexError:
            print("Unable to get FScore. Skipping.")

            return 0.0, 0.0, 0.0


    def get_score(self, test_input):
        #import time

        current_epoch = test_input.epochs

        scores = []


        #tic = time.clock()
        while test_input.epochs == current_epoch:

            (inputs_vals, predicate_vals, target_vals), seq_len_vals = test_input.next_batch(self.cfg.batch_dev_size)

            char_vals, char_lens = None, None
            if "char" in self.cfg.emb_type:
                char_vals, char_lens = self.charpp.get_data(inputs_vals)

            feeddct_err = self.get_feed(inputs_vals, predicate_vals, target_vals, seq_len_vals,
                                        self.cfg.learning_rate, char_vals, char_lens, is_train=False)



            [score] = self.sess.run(
                            [self.logits], feed_dict=feeddct_err)

            scores.extend(score.tolist())



        return scores

    def get_transition_params(self, labels):
        '''Construct transtion scoresd (0 for allowed, -inf for invalid).
        Args:
          label_strs: A [num_tags,] sequence of BIO-tags.
        Returns:
          A [num_tags, num_tags] matrix of transition scores.
        '''
        num_tags = len(labels)
        transition_params = np.zeros([num_tags, num_tags], dtype=np.float32)

        for i in range(num_tags):
            for j in range(num_tags):
                # if  index_to_labels[i][0] == 'B':
                if i != j and self.pp.labels[j][0] == 'I' and not self.pp.labels[i] == 'B' + self.pp.labels[j][1:]:
                    transition_params[i, j] = np.NINF
                if i == 0 or j == 0:
                    transition_params[i, j] = np.NINF

        return transition_params

    def inference(self, score, transition_params):
        scores = np.asarray(score)
        return tf.contrib.crf.viterbi_decode(scores, transition_params)


    def post_processing(self, seq):
        status = 0  # None
        for i in range(len(seq)):
            if seq[i][0] == "I":
                if status == 0:
                    seq[i] = "O"
                if status == 1:
                    status = 2
            if seq[i][0] == "B":
                status = 1
            if seq[i][0] == "O":
                status = 0
        return seq

    def evaluate(self, path):
        test = CoNLL2005Reader(path).read_all()
        test_data = self.pp.get_data_eval(path, test)
        test_input = PaddedDataIteratorSEQ(test_data)
        print(self.conll_evaluate(test_input, test, path))

def main(config_path="config/srl.ini"):
    tf.set_random_seed(12345)
    dsrl = DSRL(config_path)
    if dsrl.cfg.mode == "train":
        dsrl.train()
    #if dsrl.cfg.mode == "infer":
    #    print(dsrl.evaluate(dsrl.cfg.test_path))

    if dsrl.cfg.mode == "infer":
        # first, argmax inference

        print (dsrl.evaluate(dsrl.cfg.test_path))



if __name__ == '__main__':
    import sys

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[2]
    main(sys.argv[1])
