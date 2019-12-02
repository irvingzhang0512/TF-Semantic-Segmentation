import tensorflow as tf


def build_accuracy_fn(num_classes=21):
    def accuracy(y_true, y_pred):
        logits = y_pred
        labels = tf.cast(y_true, tf.int32)
        pred_classes = tf.expand_dims(
            tf.argmax(logits, axis=3, output_type=tf.int32), axis=3)

        # base results
        labels = tf.squeeze(labels, axis=-1)  # reduce the channel dimension.
        labels_flat = tf.reshape(labels, [-1, ])
        valid_indices = tf.cast(labels_flat <= num_classes - 1, tf.int32)
        valid_labels = tf.dynamic_partition(
            labels_flat, valid_indices, num_partitions=2)[1]

        # get predictions
        preds_flat = tf.reshape(pred_classes, [-1, ])
        valid_preds = tf.dynamic_partition(
            preds_flat, valid_indices, num_partitions=2)[1]

        accuracy = tf.reduce_mean(
            tf.cast(tf.equal(valid_labels, valid_preds), tf.float32))

        return accuracy

    return accuracy


def build_mean_iou_fn(num_classes=21):
    def mean_iou(y_true, y_pred):
        logits = y_pred
        labels = tf.cast(y_true, tf.int32)

        pred_classes = tf.expand_dims(
            tf.argmax(logits, axis=3, output_type=tf.int32), axis=3)
        labels = tf.squeeze(labels, axis=-1)  # reduce the channel dimension.
        labels_flat = tf.reshape(labels, [-1, ])
        valid_indices = tf.cast(labels_flat <= num_classes - 1, tf.int32)
        valid_labels = tf.dynamic_partition(
            labels_flat, valid_indices, num_partitions=2)[1]
        preds_flat = tf.reshape(pred_classes, [-1, ])
        valid_preds = tf.dynamic_partition(
            preds_flat, valid_indices, num_partitions=2)[1]

        cm = tf.math.confusion_matrix(
            valid_labels, valid_preds, num_classes
        )

        def compute_mean_iou(total_cm):
            sum_over_row = tf.cast(
                tf.reduce_sum(total_cm, 0), tf.float32)
            sum_over_col = tf.cast(
                tf.reduce_sum(total_cm, 1), tf.float32)
            cm_diag = tf.cast(tf.diag_part(total_cm), tf.float32)
            denominator = sum_over_row + sum_over_col - cm_diag

            # The mean is only computed over classes that appear in the
            # label or prediction tensor. If the denominator is 0, we need to
            # ignore the class.
            num_valid_entries = tf.reduce_sum(
                tf.cast(
                    tf.not_equal(denominator, 0), dtype=tf.float32))

            # If the value of the denominator is 0, set it to 1 to avoid
            # zero division.
            denominator = tf.where(
                tf.greater(denominator, 0), denominator,
                tf.ones_like(denominator))
            iou = tf.div(cm_diag, denominator)

            # If the number of valid entries is 0 (no classes) we return 0.
            result = tf.where(
                tf.greater(num_valid_entries, 0),
                tf.reduce_sum(iou, name='mean_iou') / num_valid_entries, 0)
            return result

        score = compute_mean_iou(cm)

        return score

    return mean_iou
