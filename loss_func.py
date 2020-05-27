import tensorflow as tf

def cross_entropy_loss(inputs, true_w):
    """
    ==========================================================================

    inputs: The embeddings for context words. Dimension is [batch_size, embedding_size].
    true_w: The embeddings for predicting words. Dimension of true_w is [batch_size, embedding_size].

    Write the code that calculate A = log(exp({u_o}^T v_c))

    A =


    And write the code that calculate B = log(\sum{exp({u_w}^T v_c)})


    B =

    ==========================================================================
    """

    #dot product of context and target
    dot_prd = tf.reduce_sum(tf.multiply(inputs, true_w), axis=1)
    dot_prd = tf.reshape(tf.where(tf.is_nan(dot_prd), tf.zeros_like(dot_prd), dot_prd), [list(inputs.get_shape())[0], 1])
    exp_dot_prod = tf.exp(dot_prd)
    A = tf.log(exp_dot_prod)

    B_prod = tf.matmul(inputs, true_w, transpose_b=True)
    exp_b_prod = tf.exp(B_prod)
    dot_prod_B = tf.reduce_sum(exp_b_prod, axis=1)
    log_B = tf.log(dot_prod_B)
    B = tf.reshape(log_B, [list(inputs.get_shape())[0], 1])

    return tf.subtract(B, A)


def nce_loss(inputs, weights, biases, labels, sample, unigram_prob):
    """
        ==========================================================================

        inputs: Embeddings for context words. Dimension is [batch_size, embedding_size].
        weigths: Weights for nce loss. Dimension is [Vocabulary, embeeding_size].
        biases: Biases for nce loss. Dimension is [Vocabulary, 1].
        labels: Word_ids for predicting words. Dimesion is [batch_size, 1].
        samples: Word_ids for negative samples. Dimension is [num_sampled].
        unigram_prob: Unigram probability. Dimesion is [Vocabulary].

        Implement Noise Contrastive Estimation Loss Here

        ==========================================================================
    """
    loss_correction_1 = []
    for i in range(list(inputs.get_shape())[0]):
        loss_correction_1.append(0.0000000001)

    loss_correction_2 = []
    for i in range(list(inputs.get_shape())[0]):
        for j in range(len(sample)):
            loss_correction_2.append(0.0000000001)

    result = tf.scalar_mul(-1, tf.add(tf.log(tf.sigmoid(tf.subtract(tf.reshape(tf.reduce_sum(tf.multiply(inputs,
                                                                                                         tf.reshape(tf.nn.embedding_lookup(weights, labels, name="T_wrd_embed"),
                                   [list(inputs.get_shape())[0], inputs.get_shape().as_list()[1]])), axis=1),
                              [list(inputs.get_shape())[0], 1]) , tf.log(
        tf.add(tf.scalar_mul(len(sample), tf.nn.embedding_lookup
        (tf.convert_to_tensor(unigram_prob, dtype=tf.float32), labels, name="T_uni_prob")),
               tf.reshape(tf.convert_to_tensor(loss_correction_1, dtype=tf.float32),
                          [list(inputs.get_shape())[0], 1]))) )))

    , tf.reshape(tf.reduce_sum(tf.log(tf.add(tf.subtract([[1.0 for j in range(len(sample))] for i in range(list(inputs.get_shape())[0])],
                                                         tf.sigmoid(tf.subtract(tf.add(tf.matmul(inputs, tf.nn.embedding_lookup(weights, tf.convert_to_tensor(sample, dtype=tf.int32),
                                                        name="N_tgt_embed"), transpose_b=True),
                                                                                       tf.tile(tf.transpose(tf.reshape(tf.nn.embedding_lookup(biases, sample, name="N_tgt_bias"),
                                      [len(sample), 1])), [list(inputs.get_shape())[0], 1])), tf.tile(tf.log(
        tf.scalar_mul(len(sample), tf.transpose(
        tf.reshape(tf.nn.embedding_lookup(tf.convert_to_tensor(unigram_prob, dtype=tf.float32), sample, name="N_uni_prob"),
                   [len(sample), 1])))), [list(inputs.get_shape())[0], 1])))), tf.reshape(tf.convert_to_tensor(loss_correction_2,
                                                                                                                   dtype=tf.float32), [list(inputs.get_shape())[0], len(sample)]))), axis=1),
                                       [list(inputs.get_shape())[0], 1])))

    return result

