import tensorflow as tf


def get_train_spec(logits, labels, optimizer):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(tf.estimator.ModeKeys.TRAIN, loss=loss, train_op=train_op)


def get_evaluation_spec(logits, labels, num_classes):
    predicted_classes = tf.argmax(logits, axis=-1)

    labels_one_hot = tf.one_hot(labels, depth=num_classes)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_one_hot))

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

    return tf.estimator.EstimatorSpec(tf.estimator.ModeKeys.EVAL,
                                      loss=loss, eval_metric_ops=metrics)


def get_predict_spec(logits, target_file_name):
    predicted_classes = tf.argmax(logits, axis=-1)
    predictions = {
        'class_ids': predicted_classes,
        'probabilities': tf.nn.softmax(logits),
        'logits': logits,
        'target_file_name': target_file_name,
    }
    return tf.estimator.EstimatorSpec(tf.estimator.ModeKeys.PREDICT,
                                      predictions=predictions)
