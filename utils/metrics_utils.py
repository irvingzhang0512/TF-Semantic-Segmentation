import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.training import distribution_strategy_context

def metric_variable(shape, dtype, validate_shape=True, name=None):
  """Create variable in `GraphKeys.(LOCAL|METRIC_VARIABLES)` collections.

  If running in a `DistributionStrategy` context, the variable will be
  "tower local". This means:

  *   The returned object will be a container with separate variables
      per replica/tower of the model.

  *   When writing to the variable, e.g. using `assign_add` in a metric
      update, the update will be applied to the variable local to the
      replica/tower.

  *   To get a metric's result value, we need to sum the variable values
      across the replicas/towers before computing the final answer.
      Furthermore, the final answer should be computed once instead of
      in every replica/tower. Both of these are accomplished by
      running the computation of the final result value inside
      `tf.contrib.distribution_strategy_context.get_tower_context(
      ).merge_call(fn)`.
      Inside the `merge_call()`, ops are only added to the graph once
      and access to a tower-local variable in a computation returns
      the sum across all replicas/towers.

  Args:
    shape: Shape of the created variable.
    dtype: Type of the created variable.
    validate_shape: (Optional) Whether shape validation is enabled for
      the created variable.
    name: (Optional) String name of the created variable.

  Returns:
    A (non-trainable) variable initialized to zero, or if inside a
    `DistributionStrategy` scope a tower-local variable container.
  """
  # Note that synchronization "ON_READ" implies trainable=False.
  return variable_scope.variable(
      lambda: array_ops.zeros(shape, dtype),
      collections=[
          ops.GraphKeys.LOCAL_VARIABLES, ops.GraphKeys.METRIC_VARIABLES
      ],
      validate_shape=validate_shape,
      synchronization=variable_scope.VariableSynchronization.ON_READ,
      aggregation=variable_scope.VariableAggregation.SUM,
      name=name)


def _safe_div(numerator, denominator, name):
  """Divides two tensors element-wise, returning 0 if the denominator is <= 0.

  Args:
    numerator: A real `Tensor`.
    denominator: A real `Tensor`, with dtype matching `numerator`.
    name: Name for the returned op.

  Returns:
    0 if `denominator` <= 0, else `numerator` / `denominator`
  """
  t = math_ops.truediv(numerator, denominator)
  zero = array_ops.zeros_like(t, dtype=denominator.dtype)
  condition = math_ops.greater(denominator, zero)
  zero = math_ops.cast(zero, t.dtype)
  return array_ops.where(condition, t, zero, name=name)


def _aggregate_across_towers(metrics_collections, metric_value_fn, *args):
  """Aggregate metric value across towers."""
  def fn(distribution, *a):
    """Call `metric_value_fn` in the correct control flow context."""
    if hasattr(distribution, '_outer_control_flow_context'):
      # If there was an outer context captured before this method was called,
      # then we enter that context to create the metric value op. If the
      # caputred context is `None`, ops.control_dependencies(None) gives the
      # desired behavior. Else we use `Enter` and `Exit` to enter and exit the
      # captured context.
      # This special handling is needed because sometimes the metric is created
      # inside a while_loop (and perhaps a TPU rewrite context). But we don't
      # want the value op to be evaluated every step or on the TPU. So we
      # create it outside so that it can be evaluated at the end on the host,
      # once the update ops have been evaluted.

      # pylint: disable=protected-access
      if distribution._outer_control_flow_context is None:
        with ops.control_dependencies(None):
          metric_value = metric_value_fn(distribution, *a)
      else:
        distribution._outer_control_flow_context.Enter()
        metric_value = metric_value_fn(distribution, *a)
        distribution._outer_control_flow_context.Exit()
        # pylint: enable=protected-access
    else:
      metric_value = metric_value_fn(distribution, *a)
    if metrics_collections:
      ops.add_to_collections(metrics_collections, metric_value)
    return metric_value

  return distribution_strategy_context.get_tower_context().merge_call(fn, *args)


def mean_per_class_accuracy(labels,
                            predictions,
                            num_classes,
                            weights=None,
                            metrics_collections=None,
                            updates_collections=None,
                            name=None):
  """Calculates the mean of the per-class accuracies.

  Calculates the accuracy for each class, then takes the mean of that.

  For estimation of the metric over a stream of data, the function creates an
  `update_op` operation that updates the accuracy of each class and returns
  them.

  If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.

  Args:
    labels: A `Tensor` of ground truth labels with shape [batch size] and of
      type `int32` or `int64`. The tensor will be flattened if its rank > 1.
    predictions: A `Tensor` of prediction results for semantic labels, whose
      shape is [batch size] and type `int32` or `int64`. The tensor will be
      flattened if its rank > 1.
    num_classes: The possible number of labels the prediction task can
      have. This value must be provided, since two variables with shape =
      [num_classes] will be allocated.
    weights: Optional `Tensor` whose rank is either 0, or the same rank as
      `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
      be either `1`, or the same as the corresponding `labels` dimension).
    metrics_collections: An optional list of collections that
      `mean_per_class_accuracy'
      should be added to.
    updates_collections: An optional list of collections `update_op` should be
      added to.
    name: An optional variable_scope name.

  Returns:
    mean_accuracy: A `Tensor` representing the mean per class accuracy.
    update_op: An operation that updates the accuracy tensor.

  Raises:
    ValueError: If `predictions` and `labels` have mismatched shapes, or if
      `weights` is not `None` and its shape doesn't match `predictions`, or if
      either `metrics_collections` or `updates_collections` are not a list or
      tuple.
    RuntimeError: If eager execution is enabled.
  """
  if context.executing_eagerly():
    raise RuntimeError('tf.metrics.mean_per_class_accuracy is not supported '
                       'when eager execution is enabled.')

  with variable_scope.variable_scope(name, 'mean_accuracy',
                                     (predictions, labels, weights)):
    labels = math_ops.to_int64(labels)

    # Flatten the input if its rank > 1.
    if labels.get_shape().ndims > 1:
      labels = array_ops.reshape(labels, [-1])

    if predictions.get_shape().ndims > 1:
      predictions = array_ops.reshape(predictions, [-1])

    # Check if shape is compatible.
    predictions.get_shape().assert_is_compatible_with(labels.get_shape())

    total = metric_variable([num_classes], dtypes.float32, name='total')
    count = metric_variable([num_classes], dtypes.float32, name='count')

    ones = array_ops.ones([array_ops.size(labels)], dtypes.float32)

    if labels.dtype != predictions.dtype:
      predictions = math_ops.cast(predictions, labels.dtype)
    is_correct = math_ops.to_float(math_ops.equal(predictions, labels))

    if weights is not None:
      if weights.get_shape().ndims > 1:
        weights = array_ops.reshape(weights, [-1])
      weights = math_ops.to_float(weights)

      is_correct *= weights
      ones *= weights

    update_total_op = state_ops.scatter_add(total, labels, ones)
    update_count_op = state_ops.scatter_add(count, labels, is_correct)

    def compute_mean_accuracy(_, count, total):
      per_class_accuracy = _safe_div(count, total, None)
      label_count = tf.bincount(tf.to_int32(total), minlength=num_classes, dtype=tf.int32)
      weights = tf.where(tf.greater(label_count, 0), tf.ones_like(label_count), tf.zeros_like(label_count))
      mean_accuracy_v = tf.div(tf.to_float(math_ops.reduce_sum(per_class_accuracy)),
                              tf.to_float(math_ops.reduce_sum(weights)), name='mean_accuracy')
      return mean_accuracy_v

    mean_accuracy_v = _aggregate_across_towers(
        metrics_collections, compute_mean_accuracy, count, total)

    update_op = _safe_div(update_count_op, update_total_op, name='update_op')
    if updates_collections:
      ops.add_to_collections(updates_collections, update_op)

    return mean_accuracy_v, update_op