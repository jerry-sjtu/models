from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
from six.moves import xrange
import numpy as np

from tensorflow.python.estimator import estimator
from tensorflow.python.estimator import model_fn
from tensorflow.python.estimator.canned import head as head_lib
from tensorflow.python.estimator.canned import optimizers
from tensorflow.python.feature_column import feature_column as feature_column_lib
from tensorflow.python.layers import core as core_layers
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops.losses import losses
from tensorflow.python.summary import summary
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import standard_ops
from tensorflow import concat
from tensorflow import scalar_mul
from tensorflow import multiply
from tensorflow import reduce_sum
from tensorflow import transpose
from tensorflow import expand_dims
from tensorflow import matmul
from tensorflow import tile
import tensorflow as tf


# The default learning rate of 0.05 is a historical artifact of the initial
# implementation, but seems a reasonable choice.
_LEARNING_RATE = 0.005


class CrossLayer(base.Layer):

    def __init__(self,
                 layer_id=0,
                 x0=None,
                 trainable=True,
                 name=None,
                 **kwargs):
        super(CrossLayer, self).__init__(trainable=trainable, name=name,
                                         activity_regularizer=None,
                                         **kwargs)
        self.layer_id = layer_id
        self.x0 = x0
        self.kernel_initializer = init_ops.glorot_uniform_initializer()
        self.bias_initializer = init_ops.zeros_initializer()

        self.use_bias = True
        self.activation = None

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        if input_shape[-1].value is None:
            raise ValueError('The last dimension of the inputs to `Dense` '
                             'should be defined. Found `None`.')
        self.input_spec = base.InputSpec(min_ndim=2,
                                         axes={-1: input_shape[-1].value})
        self.kernel = self.add_variable('kernel',
                                        shape=[input_shape[-1].value, 1],
                                        # shape=[10, input_shape[-1].value],
                                        initializer=self.kernel_initializer,
                                        regularizer=None,
                                        constraint=None,
                                        dtype=self.dtype,
                                        trainable=True)
        if self.use_bias:
            self.bias = self.add_variable('bias',
                                          shape=[input_shape[-1].value],
                                          # shape=[10],
                                          initializer=self.bias_initializer,
                                          regularizer=None,
                                          constraint=None,
                                          dtype=self.dtype,
                                          trainable=True)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs, **kwargs):
        inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
        shape = inputs.get_shape().as_list()
        if len(shape) > 2:
            # Broadcasting is required for the inputs.
            outputs = standard_ops.tensordot(inputs, self.kernel, [[len(shape) - 1],
                                                                   [0]])
            # Reshape the output back to the original ndim of the input.
            # if not context.executing_eagerly():
            #     output_shape = shape[:-1] + [self.units]
            #     outputs.set_shape(output_shape)
        else:
            # notice: batch product here
            # outputs = reduce_sum(multiply(self.x0, inputs), axis=1, keep_dims=True)
            outputs = matmul(expand_dims(self.x0, axis=2), expand_dims(inputs, axis=2), transpose_a=False, transpose_b=True)
            # the static shape
            # shape_kernel = convert_to_tensor([shape[0], 1, 1])
            # the dynamic shape
            shape_kernel = tf.convert_to_tensor([tf.shape(inputs)[0], 1, 1])
            outputs = matmul(outputs, tile(expand_dims(self.kernel, axis=0), multiples=shape_kernel))
            shape_outputs = tf.convert_to_tensor(tf.shape(inputs)[0:2])
            outputs = gen_math_ops.add(tf.reshape(outputs, shape=shape_outputs), inputs)
            outputs = nn.bias_add(outputs, self.bias)
            outputs = nn.relu(outputs)
            # outputs = gen_math_ops.mat_mul(inputs, self.kernel)
            # print(outputs)
        # if self.use_bias:
        #     outputs = nn.bias_add(outputs, self.bias)
        # if self.activation is not None:
        #     return self.activation(outputs)  # pylint: disable=not-callable
        return outputs

    def compute_output_shape(self, input_shape):
        return tensor_shape.TensorShape(input_shape)


def cross_layer(inputs, layer_id, x0, name=None, reuse=None):
    trainable = True
    layer = CrossLayer(layer_id=layer_id,
                       x0=x0,
                       trainable=trainable,
                       name=name,
                       dtype=inputs.dtype.base_dtype,
                       _scope=name,
                       _reuse=reuse)
    return layer.apply(inputs)


def _add_hidden_layer_summary(value, tag):
    summary.scalar('%s/fraction_of_zero_values' % tag, nn.zero_fraction(value))
    summary.histogram('%s/activation' % tag, value)


def _dnn_logit_fn_builder(units, hidden_units, feature_columns, activation_fn,
                          dropout, input_layer_partitioner):
    if not isinstance(units, int):
        raise ValueError('units must be an int.  Given type: {}'.format(
            type(units)))

    def dnn_logit_fn(features, mode):
        with variable_scope.variable_scope('input_from_feature_columns',
                                           values=tuple(six.itervalues(features)),
                                           partitioner=input_layer_partitioner):
            inputs = feature_column_lib.input_layer(features=features, feature_columns=feature_columns)
            dense = inputs
            cross = inputs
        for layer_id, num_hidden_units in enumerate(hidden_units):
            with variable_scope.variable_scope('dense_layer_%d' % layer_id, values=(dense,)) as hidden_layer_scope:
                dense = core_layers.dense(
                    dense,
                    units=num_hidden_units,
                    activation=activation_fn,
                    kernel_initializer=init_ops.glorot_uniform_initializer(),
                    name=hidden_layer_scope)
                if dropout is not None and mode == model_fn.ModeKeys.TRAIN:
                    dense = core_layers.dropout(dense, rate=dropout, training=True)
            _add_hidden_layer_summary(dense, hidden_layer_scope.name)

        with variable_scope.variable_scope('fm_layer', values=(cross,)) as cross_layer_scope:
            builder = feature_column_lib._LazyBuilder(features)

            cross = cross_layer(cross, layer_id, inputs, name=cross_layer_scope)
        _add_hidden_layer_summary(cross, cross_layer_scope.name)

        with variable_scope.variable_scope('logits', values=(dense,cross)) as logits_scope:
            dense_cross = concat([dense, cross], axis=1)
            logits = core_layers.dense(
                cross,
                units=1,
                activation=None,
                kernel_initializer=init_ops.glorot_uniform_initializer(),
                name=logits_scope)
        _add_hidden_layer_summary(logits, logits_scope.name)


        return logits

    return dnn_logit_fn


def _dnn_model_fn(features,
                  labels,
                  mode,
                  head,
                  hidden_units,
                  feature_columns,
                  optimizer='Adagrad',
                  activation_fn=nn.relu,
                  dropout=None,
                  input_layer_partitioner=None,
                  config=None):
  if not isinstance(features, dict):
    raise ValueError('features should be a dictionary of `Tensor`s. '
                     'Given type: {}'.format(type(features)))

  optimizer = optimizers.get_optimizer_instance(
      optimizer, learning_rate=_LEARNING_RATE)
  num_ps_replicas = config.num_ps_replicas if config else 0
  # 在tensorflow的ps架构中，ps负责存储模型的参数，worker负责使用训练数据对参数进行更新。默认情况下，tensorflow会把参数按照
  # round-robin的方式放到各个参数服务器（ps）上。
  partitioner = partitioned_variables.min_max_variable_partitioner(
      max_partitions=num_ps_replicas)
  with variable_scope.variable_scope(
      'dcn',
      values=tuple(six.itervalues(features)),
      partitioner=partitioner):
    input_layer_partitioner = input_layer_partitioner or (
        partitioned_variables.min_max_variable_partitioner(
            max_partitions=num_ps_replicas,
            min_slice_size=64 << 20))

    logit_fn = _dnn_logit_fn_builder(
        units=head.logits_dimension,
        hidden_units=hidden_units,
        feature_columns=feature_columns,
        activation_fn=activation_fn,
        dropout=dropout,
        input_layer_partitioner=input_layer_partitioner)
    logits = logit_fn(features=features, mode=mode)

    return head.create_estimator_spec(
        features=features,
        mode=mode,
        labels=labels,
        optimizer=optimizer,
        logits=logits)


class DCNClassifier(estimator.Estimator):
    """
        Deep Cross Network [KDD 2017]Deep & Cross Network for Ad Click Predictions
    """
    def __init__(
            self,
            hidden_units,
            feature_columns,
            model_dir=None,
            n_classes=2,
            weight_column=None,
            label_vocabulary=None,
            optimizer='Adagrad',
            activation_fn=nn.relu,
            dropout=None,
            input_layer_partitioner=None,
            config=None,
            warm_start_from=None,
            loss_reduction=losses.Reduction.SUM,
    ):
        if n_classes == 2:
            head = head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss(
                weight_column=weight_column,
                label_vocabulary=label_vocabulary,
                loss_reduction=loss_reduction)
        else:
            head = head_lib._multi_class_head_with_softmax_cross_entropy_loss(
                n_classes, weight_column=weight_column,
                label_vocabulary=label_vocabulary,
                loss_reduction=loss_reduction)

        def _model_fn(features, labels, mode, config):
            """Call the defined shared _dnn_model_fn."""
            return _dnn_model_fn(
                features=features,
                labels=labels,
                mode=mode,
                head=head,
                hidden_units=hidden_units,
                feature_columns=tuple(feature_columns or []),
                optimizer=optimizer,
                activation_fn=activation_fn,
                dropout=dropout,
                input_layer_partitioner=input_layer_partitioner,
                config=config)

        super(DCNClassifier, self).__init__(
            model_fn=_model_fn, model_dir=model_dir, config=config,
            warm_start_from=warm_start_from)