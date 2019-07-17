import tensorflow as tf
from utils.training_utils import get_model_learning_rate


__all__ = ['get_estimator_spec', 'get_preprocess_by_frontend', \
           'preprocess_subtract_imagenet_mean', 'preprocess_zero_to_one', 'preprocess_zero_to_one']


def get_estimator_spec(mode, logits, init_fn, labels, num_classes=None, params=None, features=None):
  """
  Reference: https://github.com/rishizek/tensorflow-deeplab-v3
  """
  pred_classes = tf.expand_dims(tf.argmax(logits, axis=3, output_type=tf.int32), axis=3)

  predictions = {
      'class_ids': pred_classes,
      'probabilities': tf.nn.softmax(logits),
      'logits': logits,
      'target_file_name': labels,
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions['preprocessed_image'] = features['image']
    predictions['label'] = features['label']
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions)

  predictions.pop('target_file_name')

  # base results
  labels = tf.squeeze(labels, axis=3)  # reduce the channel dimension.
  logits_by_num_classes = tf.reshape(logits, [-1, num_classes])
  labels_flat = tf.reshape(labels, [-1, ])
  valid_indices = tf.to_int32(labels_flat <= num_classes - 1)
  valid_logits = tf.dynamic_partition(logits_by_num_classes, valid_indices, num_partitions=2)[1]
  valid_labels = tf.dynamic_partition(labels_flat, valid_indices, num_partitions=2)[1]

  # get predictions
  preds_flat = tf.reshape(pred_classes, [-1, ])
  valid_preds = tf.dynamic_partition(preds_flat, valid_indices, num_partitions=2)[1]
  confusion_matrix = tf.confusion_matrix(valid_labels, valid_preds, num_classes=num_classes)
  predictions['valid_preds'] = valid_preds
  predictions['valid_labels'] = valid_labels
  predictions['confusion_matrix'] = confusion_matrix

  # get loss
  cross_entropy = tf.losses.sparse_softmax_cross_entropy(
      logits=valid_logits, labels=valid_labels)
  tf.identity(cross_entropy, name='cross_entropy')
  tf.summary.scalar('cross_entropy', cross_entropy)
  loss = tf.losses.get_total_loss()

  # get train_op
  if mode == tf.estimator.ModeKeys.TRAIN:
    train_op = _get_train_op(params, loss)
  else:
    train_op = None
  
  # get metrics
  accuracy = tf.metrics.accuracy(valid_labels, valid_preds)
  mean_iou = tf.metrics.mean_iou(valid_labels, valid_preds, num_classes)
  metrics = {'px_accuracy': accuracy, 'mean_iou': mean_iou}

  # get training metrics for summary & logging
  tf.identity(accuracy[1], name='train_px_accuracy')
  tf.summary.scalar('train_px_accuracy', accuracy[1])
  def compute_mean_iou(total_cm, name='mean_iou'):
    """Compute the mean intersection-over-union via the confusion matrix."""
    sum_over_row = tf.to_float(tf.reduce_sum(total_cm, 0))
    sum_over_col = tf.to_float(tf.reduce_sum(total_cm, 1))
    cm_diag = tf.to_float(tf.diag_part(total_cm))
    denominator = sum_over_row + sum_over_col - cm_diag
    num_valid_entries = tf.reduce_sum(tf.cast(
        tf.not_equal(denominator, 0), dtype=tf.float32))
    denominator = tf.where(
        tf.greater(denominator, 0),
        denominator,
        tf.ones_like(denominator))
    iou = tf.div(cm_diag, denominator)
    for i in range(num_classes):
      tf.identity(iou[i], name='train_iou_class{}'.format(i))
      tf.summary.scalar('train_iou_class{}'.format(i), iou[i])
    result = tf.where(
        tf.greater(num_valid_entries, 0),
        tf.reduce_sum(iou, name=name) / num_valid_entries,
        0)
    return result
  train_mean_iou = compute_mean_iou(mean_iou[1])
  tf.identity(train_mean_iou, name='train_mean_iou')
  tf.summary.scalar('train_mean_iou', train_mean_iou)

  # pre trained model
  scaffold = None
  if init_fn is not None:
    def _init_fn(scaffold, session):
        init_fn(session)
    scaffold = tf.train.Scaffold(init_fn=_init_fn)

  # final results
  return tf.estimator.EstimatorSpec(
    mode=mode,
    predictions=predictions,
    loss=loss,
    train_op=train_op,
    eval_metric_ops=metrics,
    scaffold=scaffold)


def _get_train_op(params, total_loss):
  lr = get_model_learning_rate(params['learning_policy'], 
                               params['base_learning_rate'],
                               params['learning_rate_decay_step'], 
                               params['learning_rate_decay_factor'],
                               params['training_number_of_steps'], 
                               params['learning_power'],
                               params['end_learning_rate'],
                               params['slow_start_step'], 
                               params['slow_start_learning_rate'])
  tf.identity(lr, name='learning_rate')
  tf.summary.scalar('learning_rate', lr)
  optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=params['momentum'])
  return optimizer.minimize(total_loss, global_step=tf.train.get_global_step())


# Mean pixel value.
_MEAN_RGB = [123.15, 115.90, 103.06]


def preprocess_subtract_imagenet_mean(inputs, dtype=tf.float32):
  """Subtract Imagenet mean RGB value."""
  mean_rgb = tf.reshape(_MEAN_RGB, [1, 1, 1, 3])
  num_channels = tf.shape(inputs)[-1]
  # We set mean pixel as 0 for the non-RGB channels.
  mean_rgb_extended = tf.concat(
      [mean_rgb, tf.zeros([1, 1, 1, num_channels - 3])], axis=3)
  return tf.cast(inputs - mean_rgb_extended, dtype=dtype)


def preprocess_zero_mean_unit_range(inputs, dtype=tf.float32):
  """Map image values from [0, 255] to [-1, 1]."""
  preprocessed_inputs = (2.0 / 255.0) * tf.to_float(inputs) - 1.0
  return tf.cast(preprocessed_inputs, dtype=dtype)

def preprocess_zero_to_one(inputs, dtype=tf.float32):
  """Map image values from [0, 255] to [0, 1]."""
  preprocessed_inputs = tf.to_float(inputs) / 255.0
  return tf.cast(preprocessed_inputs, dtype=dtype)


def get_preprocess_by_frontend(frontend):
  if 'ResNet' in frontend:
      _preprocess = preprocess_subtract_imagenet_mean
  else:
      _preprocess = preprocess_zero_mean_unit_range
  return _preprocess
