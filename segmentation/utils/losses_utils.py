import tensorflow as tf


def cross_entropy_loss(y_true, y_pred, num_classes=21):
    logits = y_pred
    labels = tf.cast(y_true, tf.int32)

    # base results
    labels = tf.squeeze(labels, axis=-1)  # reduce the channel dimension.
    logits_by_num_classes = tf.reshape(logits, [-1, num_classes])
    labels_flat = tf.reshape(labels, [-1, ])
    valid_indices = tf.to_int32(labels_flat <= num_classes - 1)
    valid_logits = tf.dynamic_partition(
        logits_by_num_classes, valid_indices, num_partitions=2)[1]
    valid_labels = tf.dynamic_partition(
        labels_flat, valid_indices, num_partitions=2)[1]

    # get loss
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(
        logits=valid_logits, labels=valid_labels)
    return cross_entropy
