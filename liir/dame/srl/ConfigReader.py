import configparser


class Configuration:
    def __init__(self, path):
        self.config = configparser.ConfigParser()
        self.config.read(path)
        self.train_path = self.get_config_attribute("data", "train")
        self.dev_path = self.get_config_attribute("data", "dev")
        self.test_brown_path = self.get_config_attribute("data", "test_brown")
        self.test_wsj_path = self.get_config_attribute("data", "test_wsj")

        self.test_path = self.get_config_attribute("data", "test")

        self.extend_vob = self.get_config_attribute("data", "extend_vob", bool, False)

        self.eval_script = self.get_config_attribute("model", "eval_script", str, "scripts/srl/run_eval.sh")

        self.type = self.get_config_attribute("model", "type", str, "softmax")

        self.emb_type = self.get_config_attribute("emb", "type", str, "word")
        self.embedding_size = self.get_config_attribute("emb", "embedding_size", int, 100)
        self.embedding_pred_size = self.get_config_attribute("emb", "embedding_pred_size", int, 100)
        self.train_word_embeddings = self.get_config_attribute("emb", "train_word_embeddings", bool, True)
        self.train_pred_embeddings = self.get_config_attribute("emb", "train_pred_embeddings", bool, True)
        self.word_embedding_path = self.get_config_attribute("emb", "word")

        self.embedding_char_size = self.get_config_attribute("emb", "embedding_char_size", int, 100)
        self.hidden_char_dim = self.get_config_attribute("emb", "hidden_char_dim", int, 200)

        self.learning_rate = self.get_config_attribute("model", "learning_rate", float, 0.01)
        self.epsilon = self.get_config_attribute("model", "epsilon", float, 1e-08)
        self.optimizer = self.get_config_attribute("model", "optimizer", default_val="adam")
        self.num_layers = self.get_config_attribute("model", "num_layers", int, 2)
        self.dropput_keep_prob = self.get_config_attribute("model", "dropout_keep_prob", float, 1.0)
        self.model_dir = self.get_config_attribute("save", "model_dir", default_val=".")

        self.nb_epochs = self.get_config_attribute("model", "nb_epochs", int, 5)
        self.batch_size = self.get_config_attribute("model", "batch_size", int, 80)
        self.batch_dev_size = self.get_config_attribute("model", "batch_dev_size", int, 500)
        self.label_smoothing = self.get_config_attribute("model", "label_smoothing", float, 0.0)
        self.smooth_rare_words = self.get_config_attribute("model", "smooth_rare_words", int, 0)

        self.hidden_dim = self.get_config_attribute("model", "hidden_dim", int, 200)
        self.model_type = self.get_config_attribute("model", "type", default_val="rnn")
        self.layer_type = self.get_config_attribute("model", "layer_type", default_val="lstm")
        self.top_type = self.get_config_attribute("model", "top", default_val="softmax")
        self.log_file = self.get_config_attribute("log", "path", default_val="log.txt")
        self.patience = self.get_config_attribute("model", "patience", int, 10)
        self.num_buckets = self.get_config_attribute("model", "num_buckets", int, 5)
        self.mode = self.get_config_attribute("session", "mode", default_val="train")
        self.tag = self.get_config_attribute("session", "tag", default_val="best")
        self.max_train_length = self.get_config_attribute("data", "max_train_length", int, 100)

        self.save_freq = self.get_config_attribute("save", "save_freq", int, 5)
        self.max_to_save = self.get_config_attribute("save", "max_to_save", int, 10)

        self.residual_dropout = self.get_config_attribute("thirdparties", "residual_dropout", float, 0.0)
        self.attention_dropout = self.get_config_attribute("thirdparties", "attention_dropout", float, 0.0)
        self.relu_dropout = self.get_config_attribute("thirdparties", "relu_dropout", float, 0.0)

        self.filter_size = self.get_config_attribute("att", "filter_size", int, 1024)

        self.filter_width = self.get_config_attribute("att", "filter_width", int, 3)

        self.num_heads = self.get_config_attribute("att", "num_heads", int, 8)

        self.layer_preprocessor = self.get_config_attribute("att", "layer_preprocessor", default_val="none")
        self.layer_postprocessor = self.get_config_attribute("att", "layer_postprocessor", default_val="layer_norm")
        self.attention_function = self.get_config_attribute("att", "attention_function", default_val="dot_product")
        self.multiply_embedding_mode = self.get_config_attribute("att", "multiply_embedding_mode",
                                                                 default_val="sqrt_depth")

        self.learning_rate_decay_start = self.get_config_attribute("model", "learning_rate_decay_start", int, 400)
        self.learning_rate_decay_batch_step = self.get_config_attribute("model", "learning_rate_decay_batch_step", int,
                                                                        100)

        self.learning_rate_decay_step = self.get_config_attribute("model", "learning_rate_decay_step", float, 0.5)

        self.inference_type = self.get_config_attribute("inference", "type", default_val="A*")

    def get_config_attribute(self, section, name, type=str, default_val=None):
        try:
            return type(self.config[section][name])
        except Exception:
            return default_val
