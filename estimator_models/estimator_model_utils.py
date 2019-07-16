import tensorflow as tf
from utils.training_utils import get_model_learning_rate


__all__ = ['get_estimator_spec', 'get_preprocess_by_frontend', \
           'preprocess_subtract_imagenet_mean', 'preprocess_zero_to_one', 'preprocess_zero_to_one']


def get_estimator_spec(mode, logits, init_fn, labels, num_classes=None, params=None):
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(tf.estimator.ModeKeys.PREDICT,
                                      predictions=_get_predictions(logits, labels))
  
  if num_classes is None:
    raise ValueError('in train/eval mode, num_classes cannot be None!')
  total_loss = _get_total_loss(logits, labels, num_classes)
  if mode == tf.estimator.ModeKeys.TRAIN:
    if params is None:
      raise ValueError('in train mode, params cannot be None!')
    scaffold = None
    if init_fn is not None:
      def _init_fn(scaffold, session):
          init_fn(session)
      scaffold = tf.train.Scaffold(init_fn=_init_fn)
    return tf.estimator.EstimatorSpec(tf.estimator.ModeKeys.TRAIN, 
                                      loss=total_loss, 
                                      train_op=_get_train_op(params, total_loss),
                                      scaffold=scaffold)
  elif mode == tf.estimator.ModeKeys.EVAL:
    return tf.estimator.EstimatorSpec(tf.estimator.ModeKeys.EVAL,
                                      loss=total_loss, 
                                      eval_metric_ops=_get_eval_metrics(logits, labels, num_classes))


def _get_total_loss(logits, labels, num_classes):
  # labels_one_hot = tf.one_hot(labels, depth=num_classes)
  # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
  labels = tf.cast(labels, tf.int32)
  logits = tf.reshape(logits, [tf.shape(logits)[0], -1, num_classes])
  labels = tf.reshape(labels, [tf.shape(labels)[0], -1])
  loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=logits, labels=labels, name="loss"))
  total_loss = tf.losses.get_regularization_loss() + loss
  return total_loss


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
  optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=params['momentum'])
  return optimizer.minimize(total_loss, global_step=tf.train.get_global_step())


def _get_eval_metrics(logits, labels, num_classes):
  predicted_classes = tf.expand_dims(tf.argmax(logits, axis=-1), axis=-1)

  label_count = tf.bincount(labels, minlength=num_classes, dtype=tf.int32)
  weights = tf.where(tf.greater(label_count, 0), tf.ones_like(label_count), tf.zeros_like(label_count))
  pixel_accuracy = tf.metrics.accuracy(labels=labels,
                                        predictions=predicted_classes,
                                        name='pixel_accuracy')
  iou = tf.metrics.mean_iou(labels=labels,
                            predictions=predicted_classes,
                            num_classes=num_classes,
                            name='mean_iou')
  
  mean_per_class_accuracy = tf.metrics.mean_per_class_accuracy(labels=labels,
                                                                predictions=predicted_classes,
                                                                num_classes=num_classes,
                                                                name='mean_per_class_accuracy')

  metrics = {
      'pixel_accuracy': pixel_accuracy,
      'mean_iou': iou,
      'mean_per_class_accuracy': mean_per_class_accuracy,
  }

  return metrics


def _get_predictions(logits, target_file_name):
  predicted_classes = tf.argmax(logits, axis=-1)
  predictions = {
      'class_ids': predicted_classes,
      'probabilities': tf.nn.softmax(logits),
      'logits': logits,
      'target_file_name': target_file_name,
  }
  return predictions

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
