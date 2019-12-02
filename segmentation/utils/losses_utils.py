import tensorflow as tf


def build_cross_entropy_loss_fn(num_classes):
    def _cross_entropy_loss(y_true, y_pred):
        logits = y_pred
        labels = tf.cast(y_true, tf.int32)

        # base results
        labels = tf.squeeze(labels, axis=-1)  # reduce the channel dimension.
        logits_by_num_classes = tf.reshape(logits, [-1, num_classes])
        labels_flat = tf.reshape(labels, [-1, ])
        valid_indices = tf.cast(labels_flat <= num_classes - 1, tf.int32)
        valid_logits = tf.dynamic_partition(
            logits_by_num_classes, valid_indices, num_partitions=2)[1]
        valid_labels = tf.dynamic_partition(
            labels_flat, valid_indices, num_partitions=2)[1]

        # get loss
        cross_entropy = tf.compat.v1.losses.sparse_softmax_cross_entropy(
            logits=valid_logits, labels=valid_labels)
        return cross_entropy
    return _cross_entropy_loss


def build_total_loss_fn(num_classes, trainable_variables, weight_decay):
    def total_loss(y_true, y_pred):
        # cross_entropy
        logits = y_pred
        labels = tf.cast(y_true, tf.int32)
        labels = tf.squeeze(labels, axis=-1)  # reduce the channel dimension.
        logits_by_num_classes = tf.reshape(logits, [-1, num_classes])
        labels_flat = tf.reshape(labels, [-1, ])
        valid_indices = tf.cast(labels_flat <= num_classes - 1, tf.int32)
        valid_logits = tf.dynamic_partition(
            logits_by_num_classes, valid_indices, num_partitions=2)[1]
        valid_labels = tf.dynamic_partition(
            labels_flat, valid_indices, num_partitions=2)[1]
        cross_entropy = tf.compat.v1.losses.sparse_softmax_cross_entropy(
            logits=valid_logits, labels=valid_labels)

        # l2 loss
        l2_loss = tf.add_n(
            [tf.nn.l2_loss(v) * weight_decay for v in trainable_variables])

        return cross_entropy + l2_loss

    return total_loss
