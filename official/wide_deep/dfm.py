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
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops.losses import losses
from tensorflow.python.summary import summary
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
_LEARNING_RATE = 0.05


def _add_hidden_layer_summary(value, tag):
    summary.scalar('%s/fraction_of_zero_values' % tag, nn.zero_fraction(value))
    summary.histogram('%s/activation' % tag, value)


def _dfm_logit_fn_builder(units, hidden_units, activation_fn,
                          linear_feature_columns, dnn_feature_columns, fm_feature_columns,
                          dropout, input_layer_partitioner):
    if not isinstance(units, int):
        raise ValueError('units must be an int.  Given type: {}'.format(
            type(units)))

    def dnn_logit_fn(features, mode):
        with variable_scope.variable_scope('input_from_feature_columns',
                                           values=tuple(six.itervalues(features)),
                                           partitioner=input_layer_partitioner):
            inputs = feature_column_lib.input_layer(features=features, feature_columns=dnn_feature_columns)
            dense = inputs
        # dnn logits
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
        with variable_scope.variable_scope('dnn_logits', values=(dense,)) as logits_scope:
            dnn_logits = core_layers.dense(
                dense,
                units=units,
                activation=None,
                kernel_initializer=init_ops.glorot_uniform_initializer(),
                name=logits_scope)
        _add_hidden_layer_summary(dnn_logits, logits_scope.name)

        # linear logits
        # with variable_scope.variable_scope(
        #         'linear_logits',
        #         values=tuple(six.itervalues(features)),
        #         partitioner=input_layer_partitioner) as scope:
        #     linear_logits = feature_column_lib.linear_model(
        #         features=features,
        #         feature_columns=linear_feature_columns,
        #         units=units,
        #         cols_to_vars={}
        #     )

        # with variable_scope.variable_scope('fm_layer', values=(inputs,)) as cross_layer_scope:
        #     builder = feature_column_lib._LazyBuilder(features)
        #     fm_outputs = []
        #     for col_pair in fm_feature_columns:
        #         column1, column2 = col_pair
        #         tensor1 = column1._get_dense_tensor(builder, trainable=True)
        #         num_elements = column1._variable_shape.num_elements()
        #         batch_size = array_ops.shape(tensor1)[0]
        #         tensor2 = column2._get_dense_tensor(builder, trainable=True)
        #         tensor1 = array_ops.reshape(tensor1, shape=(batch_size, num_elements))
        #         tensor2 = array_ops.reshape(tensor2, shape=(batch_size, num_elements))
        #         fm_outputs.append(matmul(tensor1, tensor2))
        #     fm_outputs = tf.convert_to_tensor(fm_outputs)
        # _add_hidden_layer_summary(fm_outputs, cross_layer_scope.name)

        # with variable_scope.variable_scope('logits', values=(dnn_logits, linear_logits)) as logits_scope:
        #     logits = dnn_logits + linear_logits
        # _add_hidden_layer_summary(logits, logits_scope.name)
        return dnn_logits

    return dnn_logit_fn


def _dfm_model_fn(features,
                  labels,
                  mode,
                  head,
                  hidden_units,
                  linear_feature_columns,
                  dnn_feature_columns,
                  fm_feature_columns,
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

    logit_fn = _dfm_logit_fn_builder(
        units=head.logits_dimension,
        hidden_units=hidden_units,
        linear_feature_columns=linear_feature_columns,
        dnn_feature_columns=dnn_feature_columns,
        fm_feature_columns=fm_feature_columns,
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


class DeepFMClassifier(estimator.Estimator):
    """
        Deep Cross Network [KDD 2017]Deep & Cross Network for Ad Click Predictions
    """
    def __init__(
            self,
            dnn_hidden_units,
            linear_feature_columns,
            dnn_feature_columns,
            fm_feature_columns,
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
            return _dfm_model_fn(
                features=features,
                labels=labels,
                mode=mode,
                head=head,
                hidden_units=dnn_hidden_units,
                dnn_feature_columns=dnn_feature_columns,
                linear_feature_columns=linear_feature_columns,
                fm_feature_columns=fm_feature_columns,
                optimizer=optimizer,
                activation_fn=activation_fn,
                dropout=dropout,
                input_layer_partitioner=input_layer_partitioner,
                config=config)

        super(DeepFMClassifier, self).__init__(
            model_fn=_model_fn, model_dir=model_dir, config=config,
            warm_start_from=warm_start_from)