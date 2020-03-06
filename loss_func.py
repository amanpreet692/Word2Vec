import tensorflow as tf


def cross_entropy_loss(inputs, true_w):
    """
    ==========================================================================

    inputs: The embeddings for context words. Dimension is [batch_size, embedding_size].
    true_w: The embeddings for predicting words. Dimension of true_w is [batch_size, embedding_size].

    Write the code that calculate A = log(exp({u_o}^T v_c))"""
    mat_mul = tf.matmul(inputs, tf.transpose(true_w)) #batch_size,batch_size
    A = tf.diag_part(mat_mul)
    A = tf.reshape(A, [tf.shape(A)[0],1])

    # And write the code that calculate B = log(\sum{exp({u_w}^T v_c)})

    B = tf.reduce_logsumexp(mat_mul, 1)
    B = tf.reshape(B, [tf.shape(B)[0], 1])
    """
    ==========================================================================
    """
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
    label_vectors = tf.nn.embedding_lookup(weights, labels)
    label_vectors = tf.reshape(label_vectors, [-1, tf.shape(inputs)[1]])  # batch_size,embedding_size
    num_sampled = tf.cast(tf.shape(sample), tf.float32)
    pos_embeddings = tf.diag_part(tf.matmul(inputs, tf.transpose(label_vectors)))
    pos_embeddings = tf.reshape(pos_embeddings,
                                [tf.shape(pos_embeddings)[0], 1])  # batch_size,batch_size --> batch_size,1
    pos_sample_prob = tf.add(pos_embeddings, tf.nn.embedding_lookup(biases, labels))
    sampling_term = tf.log(num_sampled * tf.gather(unigram_prob, labels))
    final_positive_sampling = tf.log(tf.math.sigmoid(tf.subtract(pos_sample_prob, sampling_term)) + 1.e-17)

    sampled_vectors = tf.nn.embedding_lookup(weights, sample)
    sampled_vectors = tf.reshape(sampled_vectors, [-1, tf.shape(inputs)[1]])
    neg_sample_prob = tf.matmul(inputs, tf.transpose(sampled_vectors)) + tf.nn.embedding_lookup(biases,
                                                                                                sample)  # batch_size,num_sampled

    sampling_term = tf.log(num_sampled * tf.gather(unigram_prob, sample))
    neg_sample_sigmoid = tf.math.sigmoid(neg_sample_prob - sampling_term)
    final_negative_sampling = tf.reduce_sum(tf.log(1 - neg_sample_sigmoid + 1.e-17), 1)  # batch_size,num_sampled --> batch_size,1
    final_negative_sampling = tf.reshape(final_negative_sampling, [tf.shape(final_negative_sampling)[0], 1])

    loss = (-1) * tf.add(final_positive_sampling, final_negative_sampling)
    return loss
