

import tensorflow as tf
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import RNNCell, _Linear
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.rnn_cell_impl import LSTMStateTuple, DropoutWrapper
from tensorflow.python.util import nest

__author__ = "Quynh Do"
__copyright__ = "Copyright 2017, DAME"

# this class contains some new classes for tensorflow


class ExtHighWayLSTMCell(RNNCell):
    # reference -- Srivastava et al., 2015
    def __init__(self, num_units,
                 use_peepholes=False, cell_clip=None,
                 initializer=None, num_proj=None, proj_clip=None,
                 num_unit_shards=None, num_proj_shards=None,
                 forget_bias=1.0, state_is_tuple=True,
                 activation=None, reuse=None):

        super(ExtHighWayLSTMCell, self).__init__(_reuse=reuse)
        if not state_is_tuple:
            tf.logging.warn("%s: Using a concatenated state is slower and will soon be "
                            "deprecated.  Use state_is_tuple=True.", self)
        if num_unit_shards is not None or num_proj_shards is not None:
            tf.logging.warn(
                "%s: The num_unit_shards and proj_unit_shards parameters are "
                "deprecated and will be removed in Jan 2017.  "
                "Use a variable scope with a partitioner instead.", self)

        self._num_units = num_units
        self._use_peepholes = use_peepholes
        self._cell_clip = cell_clip
        self._initializer = initializer
        self._num_proj = num_proj
        self._proj_clip = proj_clip
        self._num_unit_shards = num_unit_shards
        self._num_proj_shards = num_proj_shards
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation or math_ops.tanh

        if num_proj:
            self._state_size = (
                LSTMStateTuple(num_units, num_proj)
                if state_is_tuple else num_units + num_proj)
            self._output_size = num_proj
        else:
            self._state_size = (
                LSTMStateTuple(num_units, num_units)
                if state_is_tuple else 2 * num_units)
            self._output_size = num_units
        self._linear1 = None
        self._linear2 = None
        if self._use_peepholes:
            self._w_f_diag = None
            self._w_i_diag = None
            self._w_o_diag = None

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def __call__(self, inputs, state, scope=None):

        num_proj = self._num_units if self._num_proj is None else self._num_proj
        sigmoid = math_ops.sigmoid

        if self._state_is_tuple:
            (c_prev, m_prev) = state
        else:
            c_prev = array_ops.slice(state, [0, 0], [-1, self._num_units])
            m_prev = array_ops.slice(state, [0, self._num_units], [-1, num_proj])

        dtype = inputs.dtype
        input_size = inputs.get_shape().with_rank(2)[1]

        if input_size.value is None:
            raise ValueError("Could not infer input size from inputs.get_shape()[-1]")
        if self._linear1 is None:
            scope = vs.get_variable_scope()
            with vs.variable_scope(
                    scope, initializer=self._initializer) as unit_scope:
                if self._num_unit_shards is not None:
                    unit_scope.set_partitioner(
                        partitioned_variables.fixed_size_partitioner(
                            self._num_unit_shards))
                self._linear1 = _Linear([inputs, m_prev], 5 * self._num_units, True)

                # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        lstm_matrix = self._linear1([inputs, m_prev])

        i, j, f, o, g = array_ops.split(
            value=lstm_matrix, num_or_size_splits=5, axis=1)

        with tf.variable_scope("highway"):

            k = array_ops.split(value=_linear([inputs], self._num_units, True), num_or_size_splits=1, axis=1)

        # Diagonal connections
        if self._use_peepholes and not self._w_f_diag:
            scope = vs.get_variable_scope()
            with vs.variable_scope(
                    scope, initializer=self._initializer) as unit_scope:
                with vs.variable_scope(unit_scope):
                    self._w_f_diag = vs.get_variable(
                        "w_f_diag", shape=[self._num_units], dtype=dtype)
                    self._w_i_diag = vs.get_variable(
                        "w_i_diag", shape=[self._num_units], dtype=dtype)
                    self._w_o_diag = vs.get_variable(
                        "w_o_diag", shape=[self._num_units], dtype=dtype)

        if self._use_peepholes:
            c = (sigmoid(f + self._forget_bias + self._w_f_diag * c_prev) * c_prev +
                 sigmoid(i + self._w_i_diag * c_prev) * self._activation(j))
        else:
            c = (sigmoid(f + self._forget_bias) * c_prev + sigmoid(i) *
                 self._activation(j))

        if self._cell_clip is not None:
            # pylint: disable=invalid-unary-operand-type
            c = clip_ops.clip_by_value(c, -self._cell_clip, self._cell_clip)
            # pylint: enable=invalid-unary-operand-type
        if self._use_peepholes:
            sigmoidg = sigmoid(g)
            su = 1. - sigmoidg

            m = sigmoidg * sigmoid(o + self._w_o_diag * c) * self._activation(c) + tf.squeeze(su * k, axis=0)
        else:
            sigmoidg = sigmoid(g)
            su = 1. - sigmoidg
            m = sigmoidg * sigmoid(o) * self._activation(c) + tf.squeeze(su * k, axis=0)

        if self._num_proj is not None:
            if self._linear2 is None:
                scope = vs.get_variable_scope()
                with vs.variable_scope(scope, initializer=self._initializer):
                    with vs.variable_scope("projection") as proj_scope:
                        if self._num_proj_shards is not None:
                            proj_scope.set_partitioner(
                                partitioned_variables.fixed_size_partitioner(
                                    self._num_proj_shards))
                        self._linear2 = _Linear(m, self._num_proj, False)
            m = self._linear2(m)

            if self._proj_clip is not None:
                # pylint: disable=invalid-unary-operand-type
                m = clip_ops.clip_by_value(m, -self._proj_clip, self._proj_clip)
                # pylint: enable=invalid-unary-operand-type

        new_state = (LSTMStateTuple(c, m) if self._state_is_tuple else
                     array_ops.concat([c, m], 1))

        return m, new_state


class ExtDropoutWrapper(DropoutWrapper):
    """Operator adding dropout to inputs and outputs of the given cell."""
    # this wrapper contains one more parameter is_train
    # if is_train=0, then w = keep_prob * w , no mask

    def __init__(self, cell, input_keep_prob=1.0, output_keep_prob=1.0,
                 state_keep_prob=1.0, variational_recurrent=False,
                 input_size=None, dtype=None, seed=None,
                 dropout_state_filter_visitor=None, is_train=True):
        DropoutWrapper.__init__(self, cell, input_keep_prob, output_keep_prob,
                                state_keep_prob, variational_recurrent,
                                input_size, dtype, seed, dropout_state_filter_visitor)
        self.is_train = is_train

    def __call__(self, inputs, state, scope=None):
        """Run the cell with the declared dropouts."""

        def _should_dropout(p):
            return (not isinstance(p, float)) or p < 1

        if _should_dropout(self._input_keep_prob):

            inputs = tf.cond(self.is_train, lambda : self._dropout(inputs, "input",
                                       self._recurrent_input_noise,
                                       self._input_keep_prob), lambda : inputs * self._input_keep_prob)

        output, new_state = self._cell(inputs, state, scope)
        if _should_dropout(self._state_keep_prob):
            # Identify which subsets of the state to perform dropout on and
            # which ones to keep.

            shallow_filtered_substructure = nest.get_traverse_shallow_structure(
                self._dropout_state_filter, new_state)

            new_state = tf.cond(self.is_train, lambda : self._dropout(new_state, "state",
                                          self._recurrent_state_noise,
                                          self._state_keep_prob,
                                          shallow_filtered_substructure), lambda : new_state * self._state_keep_prob)

        if _should_dropout(self._output_keep_prob):
            output =  tf.cond(self.is_train, lambda : self._dropout(output, "output",
                                       self._recurrent_output_noise,
                                                                     self._output_keep_prob),lambda:  output * self._output_keep_prob  )

        return output, new_state


def _linear(args, output_size, bias, bias_start=0.0, scope=None):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
    Args:
      args: a 2D Tensor or a list of 2D, batch x n, Tensors.
      output_size: int, second dimension of W[i].
      bias: boolean, whether to add a bias term or not.
      bias_start: starting value to initialize the bias; 0 by default.
      scope: VariableScope for the created subgraph; defaults to "Linear".
    Returns:
      A 2D Tensor with shape [batch x output_size] equal to
      sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
    Raises:
      ValueError: if some of the arguments has unspecified or wrong shape.
    """
    if args is None or (isinstance(args, (list, tuple)) and not args):
        raise ValueError("`args` must be specified")
    if not isinstance(args, (list, tuple)):
        args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 2:
            raise ValueError(
                "Linear is expecting 2D arguments: %s" % str(shapes))
        if not shape[1]:
            raise ValueError(
                "Linear expects shape[1] of arguments: %s" % str(shapes))
        else:
            total_arg_size += shape[1]

    # Now the computation.
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [total_arg_size, output_size])
        if len(args) == 1:
            res = tf.matmul(args[0], matrix)
        else:
            res = tf.matmul(tf.concat(1, args), matrix)
        if not bias:
            return res
        bias_term = tf.get_variable(
            "Bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
    return res + bias_term



