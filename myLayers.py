import tensorflow as tf
from tensorflow import keras
import numpy as np


class CustomAttention(keras.layers.Layer):
    """Custom layer that implements attention"""
    def __init__(self, return_scores=False, **kwargs):
        if 'trainable' not in kwargs:
            kwargs['trainable'] = True
        super(CustomAttention, self).__init__(**kwargs)
        self.return_scores = return_scores
        self.trainable = kwargs['trainable']

    def build(self, input_shape):
        self.wy = self.add_weight(
            name='wy',
            shape=(input_shape[0][-1], input_shape[0][-1]),
            initializer="random_normal",
            trainable=self.trainable,
            regularizer=keras.regularizers.l2(l=4e-6)
        )

        self.w = self.add_weight(
            name='w',
            shape=(input_shape[0][-1],),
            initializer="random_normal",
            trainable=self.trainable,
            regularizer=keras.regularizers.l2(l=4e-6)
        )

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'return_scores': self.return_scores
        })
        return config

    def call(self, inputs, mask=None):
        value, key = inputs

        # Tile all weights to match batch size for matmul
        multiples = tf.concat([tf.expand_dims(tf.shape(value)[0], -1), tf.constant([1, 1])], axis=0)
        w = tf.tile(
            tf.expand_dims(
                tf.expand_dims(self.w, 1), 0
            ), multiples
        )
        wy = tf.tile(tf.expand_dims(self.wy, 0), multiples)

        y = tf.tanh(tf.matmul(key, wy))

        a = tf.matmul(y, w)
        # Make all scores of masked embeddings equal to -inf in order to make softmax zero
        a = tf.ragged.boolean_mask(a, tf.expand_dims(mask[0], -1)).to_tensor(default_value=-1e10, shape=tf.shape(a))
        a = tf.nn.softmax(a, axis=1)

        r = tf.matmul(value, a, transpose_a=True)
        if self.return_scores:
            return [r[..., 0], a]
        else:
            return r[..., 0]


class Projection(keras.layers.Layer):
    """Custom layer that implements projection"""

    def __init__(self, **kwargs):
        if 'trainable' not in kwargs:
            kwargs['trainable'] = True
        super(Projection, self).__init__(**kwargs)
        self.trainable = kwargs['trainable']

    def build(self, input_shape):
        self.wp = self.add_weight(
            name='wp',
            shape=(input_shape[0][-1], input_shape[0][-1]),
            initializer="random_normal",
            trainable=self.trainable,
            regularizer=keras.regularizers.l2(l=4e-6)
        )

        self.wx = self.add_weight(
            name='wx',
            shape=(input_shape[0][-1], input_shape[0][-1]),
            initializer="random_normal",
            trainable=self.trainable,
            regularizer=keras.regularizers.l2(l=4e-6)
        )

    def get_config(self):
        config = super().get_config().copy()
        return config

    def call(self, inputs):
        r, h = inputs

        # Tile all weights to match batch size for matmul
        multiples = tf.concat([tf.expand_dims(tf.shape(r)[0], -1), tf.constant([1, 1])], axis=0)
        wp = tf.tile(tf.expand_dims(self.wp, 0), multiples)
        wx = tf.tile(tf.expand_dims(self.wx, 0), multiples)

        # Expand vectors to matrices with singular last dimension
        r = tf.expand_dims(r, -1)
        h = tf.expand_dims(h, -1)
        return tf.tanh(tf.matmul(wp, r) + tf.matmul(wx, h))[..., 0]


class MaskSum(keras.layers.Layer):
    """Custom layer that implements sum of non-masked embeddings"""
    def __init__(self, **kwargs):
        if 'trainable' not in kwargs:
            kwargs['trainable'] = False
        super(MaskSum, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        return config

    def call(self, inputs, mask=None):
        return tf.math.reduce_sum(
            tf.ragged.boolean_mask(inputs, mask), axis=1
        )


def circulant(elements):
    """Function that generates circulant matrix from 1D tensor"""
    dim = tf.size(elements)
    range_t = tf.range(dim)
    tiled = tf.reshape(tf.tile(tf.reshape(elements, [-1, 1]), [1, dim]), [-1])
    row_indices = tf.tile(range_t, [dim])
    col_indices = tf.math.floormod(tf.reshape(tf.reshape(range_t, [-1, 1]) + tf.reshape(
        range_t, [1, -1]), [-1]), dim)
    indices = tf.stack([row_indices, col_indices], axis=1)
    return tf.scatter_nd(indices=indices, updates=tiled, shape=[dim, dim])


class WordAspectFusion(keras.layers.Layer):
    """Custom layer that implements convolution of aspect and hidden states"""
    def __init__(self, go_backwards=False, **kwargs):
        if 'trainable' not in kwargs:
            kwargs['trainable'] = False
        super(WordAspectFusion, self).__init__(**kwargs)
        self.supports_masking = True
        # this attribute determines whether to process embeddings in reverse order (for backwards-LSTM)
        self.go_backwards = go_backwards

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'go_backwards': self.go_backwards
        })
        return config

    def call(self, inputs):
        embeddings, aspect = inputs
        if self.go_backwards:
            embeddings = embeddings[:, ::-1, :]

        # normalize inputs
        embeddings /= tf.norm(embeddings, axis=2, keepdims=True)
        aspect /= tf.norm(aspect, axis=1, keepdims=True)
        # in case of zero norm, replace NaNs that are produced by division with zeros
        embeddings = tf.where(tf.math.is_nan(embeddings), tf.zeros_like(embeddings), embeddings)
        aspect = tf.where(tf.math.is_nan(aspect), tf.zeros_like(aspect), aspect)

        circulant_a = tf.map_fn(circulant, aspect)
        # reduce dimensions of circulant matrix to match dimension of embeddings
        circulant_a = circulant_a[:, :embeddings.shape[-1], :embeddings.shape[-1]]
        return tf.matmul(embeddings, circulant_a, transpose_b=True)


class SelfAttention(keras.layers.Layer):
    def __init__(self, hidden_dim=64, return_attention_scores=False, **kwargs):
        if 'trainable' not in kwargs:
            kwargs['trainable'] = True
        super(SelfAttention, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.trainable = kwargs['trainable']
        self.return_attention_scores = return_attention_scores
        self.attention = keras.layers.Attention()

    def build(self, input_shape):
        print(input_shape)
        self.wq = self.add_weight(
            name='wq',
            shape=(input_shape[-1], self.hidden_dim),
            initializer="random_normal",
            trainable=self.trainable,
            regularizer=keras.regularizers.l2(l=4e-6)
        )

        self.wk = self.add_weight(
            name='wk',
            shape=(input_shape[-1], self.hidden_dim),
            initializer="random_normal",
            trainable=self.trainable,
            regularizer=keras.regularizers.l2(l=4e-6)
        )

        self.wv = self.add_weight(
            name='wv',
            shape=(input_shape[-1], self.hidden_dim),
            initializer="random_normal",
            trainable=self.trainable,
            regularizer=keras.regularizers.l2(l=4e-6)
        )

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'hidden_dim': self.hidden_dim,
            'return_attention_scores': self.return_attention_scores
        })
        return config

    def call(self, inputs, mask=None):
        # Tile all weights to match batch size for matmul
        multiples = tf.concat([tf.expand_dims(tf.shape(inputs)[0], -1), tf.constant([1, 1])], axis=0)

        wq = tf.tile(tf.expand_dims(self.wq, 0), multiples)
        wk = tf.tile(tf.expand_dims(self.wq, 0), multiples)
        wv = tf.tile(tf.expand_dims(self.wq, 0), multiples)

        q = tf.matmul(inputs, wq)
        k = tf.matmul(inputs, wk)
        v = tf.matmul(inputs, wv)

        output = self.attention([q, k, v], mask=[None, mask], return_attention_scores=self.return_attention_scores)

        if self.return_attention_scores:
            return output[0][:, 0, :], output[1]
        else:
            return output[:, 0, :]
