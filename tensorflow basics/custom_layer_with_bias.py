import tensorflow as tf


class Linear(tf.keras.layers.Layer):

  def __init__(self, units, input_dim):
    super(Linear, self).__init__()
    w_init = tf.random_normal_initializer()
    self.w = tf.Variable(initial_value=w_init(shape=(input_dim, units),
                                              dtype='float32'),
                         trainable=True)
    b_init = tf.zeros_initializer()
    self.b = tf.Variable(initial_value=b_init(shape=(units,),
                                              dtype='float32'),
                         trainable=True)

  def call(self, inputs):
    return tf.matmul(inputs, self.w) + self.b


linear_layer = Linear(4,2)
print(linear_layer(tf.ones((2, 2))))
print(linear_layer.w)
print(linear_layer.b)
'''
tf.Tensor(
[[-0.04408899 -0.02287596  0.04922538  0.01446066]
 [-0.04408899 -0.02287596  0.04922538  0.01446066]], shape=(2, 4), dtype=float32)
<tf.Variable 'Variable:0' shape=(2, 4) dtype=float32, numpy=
array([[-0.01895243, -0.01612992,  0.07410102,  0.01618787],
       [-0.02513656, -0.00674605, -0.02487565, -0.00172721]],
      dtype=float32)>
<tf.Variable 'Variable:0' shape=(4,) dtype=float32, numpy=array([0., 0., 0., 0.], dtype=float32)>
'''