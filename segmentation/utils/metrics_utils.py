import tensorflow as tf


def accuracy(y_true, y_pred, num_classes=21):
    logits = y_pred
    labels = tf.cast(y_true, tf.int32)
    pred_classes = tf.expand_dims(
        tf.argmax(logits, axis=3, output_type=tf.int32), axis=3)

    # base results
    labels = tf.squeeze(labels, axis=-1)  # reduce the channel dimension.
    labels_flat = tf.reshape(labels, [-1, ])
    valid_indices = tf.to_int32(labels_flat <= num_classes - 1)
    valid_labels = tf.dynamic_partition(
        labels_flat, valid_indices, num_partitions=2)[1]

    # get predictions
    preds_flat = tf.reshape(pred_classes, [-1, ])
    valid_preds = tf.dynamic_partition(
        preds_flat, valid_indices, num_partitions=2)[1]

    accuracy = tf.reduce_mean(
        tf.cast(tf.equal(valid_labels, valid_preds), tf.float32))

    return accuracy


def mean_iou(y_true, y_pred, num_classes=21):
    logits = y_pred
    labels = tf.cast(y_true, tf.int32)

    pred_classes = tf.expand_dims(
        tf.argmax(logits, axis=3, output_type=tf.int32), axis=3)
    labels = tf.squeeze(labels, axis=-1)  # reduce the channel dimension.
    labels_flat = tf.reshape(labels, [-1, ])
    valid_indices = tf.to_int32(labels_flat <= num_classes - 1)
    valid_labels = tf.dynamic_partition(
        labels_flat, valid_indices, num_partitions=2)[1]
    preds_flat = tf.reshape(pred_classes, [-1, ])
    valid_preds = tf.dynamic_partition(
        preds_flat, valid_indices, num_partitions=2)[1]

    score, up_opt = tf.metrics.mean_iou(valid_labels, valid_preds, num_classes)

    tf.keras.backend.get_session().run(tf.local_variables_initializer())
    with tf.control_dependencies([up_opt]):
        score = tf.identity(score)

    return score
