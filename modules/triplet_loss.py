import tensorflow as tf
from tensorflow import keras  # type: ignore


class TripletSemiHardLoss(keras.losses.Loss):
    """
    Computes the triplet loss with semi-hard negative mining.

    The loss encourages the positive pair (anchor, positive) to be closer than any
    negative pair (anchor, negative) by at least the margin.

    Formula: max(d(A, P) - d(A, N) + margin, 0)

    This implementation handles the mining ONLINE within the batch.
    """

    def __init__(self, margin=1.0, name="triplet_semi_hard_loss"):
        super().__init__(name=name)
        self.margin = margin

    def call(self, y_true, y_pred):
        """
        y_true: Labels (Batch_Size, ) - Used to determine which pairs are same/diff class
        y_pred: Embeddings (Batch_Size, Embed_Dim) - The output of the model
        """

        # 1. Normalize embeddings (Crucial for Euclidean distance stability)
        # Although model usually does this, we ensure it here or cast types
        embeddings = tf.cast(y_pred, tf.float32)
        embeddings = tf.math.l2_normalize(embeddings, axis=1)

        # 2. Compute Pairwise Distances (Squared Euclidean)
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2<a, b>
        # Since vectors are normalized, ||a||^2 = 1
        # Dist = 2 - 2<a, b>
        dot_product = tf.matmul(embeddings, embeddings, transpose_b=True)

        # Euclidean distance matrix
        # (Using standard formula: dist_sq = sum(x**2) + sum(y**2) - 2x.y)
        square_norm = tf.linalg.diag_part(dot_product)
        distances_sq = (
            tf.expand_dims(square_norm, 1)
            - 2.0 * dot_product
            + tf.expand_dims(square_norm, 0)
        )

        # Ensure non-negative (numerical stability)
        distances_sq = tf.maximum(distances_sq, 0.0)
        distances = tf.sqrt(
            distances_sq + 1e-16
        )  # Add epsilon to avoid sqrt(0) gradient NaN

        # 3. Create Masks
        labels = tf.cast(tf.reshape(y_true, [-1]), tf.int32)

        # Adjacency matrix of labels (1 if same class, 0 if different)
        labels_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

        # Mask for Positive pairs (Same label, but not the same instance)
        mask_pos = tf.cast(labels_equal, tf.float32) - tf.eye(tf.shape(labels)[0])

        # Mask for Negative pairs (Different label)
        mask_neg = tf.cast(tf.logical_not(labels_equal), tf.float32)

        # 4. Mine Triplets (Semi-Hard)
        # We want Negatives where: Dist(A, P) < Dist(A, N) < Dist(A, P) + Margin

        # Get distances for positive pairs
        pos_dist = tf.expand_dims(distances * mask_pos, 2)

        # Get distances for negative pairs
        neg_dist = tf.expand_dims(distances * mask_neg, 1)

        # loss = max(pos_dist - neg_dist + margin, 0)

        # However, standard semi-hard mining usually picks the "hardest" negative
        # that is still "semi-hard" or just the hardest negative.
        # This notebook implementation simplifies it by computing loss for valid negatives

        # TensorFlow Addons implementation logic simplified:
        # Finding the hardest positive for each anchor
        hardest_pos_dist = tf.reduce_max(
            pos_dist, axis=2
        )  # (Batch, Batch) -> (Batch, 1)? No, logic above is complex.

        # Let's use the Notebook's exact logic which relies on broadcasting

        # Notebook logic verification:
        # valid_neg = neg_dist < pos_dist + margin AND neg_dist > pos_dist

        valid_neg = tf.logical_and(
            tf.greater(neg_dist, pos_dist), tf.less(neg_dist, pos_dist + self.margin)
        )

        # If no semi-hard negative exists, we usually pick the hardest negative
        # Or in this code: they use a large value to mask out invalids

        semi_hard_neg = tf.where(valid_neg, neg_dist, tf.ones_like(neg_dist) * 1e12)

        # Find the hardest negative (smallest distance) among the semi-hards
        hardest_neg = tf.reduce_min(semi_hard_neg, axis=2)

        # Find hardest positive (largest distance)
        hardest_pos = tf.reduce_max(pos_dist, axis=2)  # Actually pos_dist is (B, B, 1)

        # Calculate Loss
        # We aggregate over the batch
        loss = tf.maximum(hardest_pos - hardest_neg + self.margin, 0.0)

        return tf.reduce_mean(loss)
