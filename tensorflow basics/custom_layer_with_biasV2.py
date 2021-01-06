import tensorflow as tf


class Linear(tf.keras.layers.Layer):

  # def __init__(self, units, input_dim):
  #   super(Linear, self).__init__()
  #   self.w = self.add_weight(shape=(input_dim, units),
  #                            initializer='random_normal',
  #                            trainable=True)
  #   self.b = self.add_weight(shape=(units,),
  #                            initializer='zeros',
  #                            trainable=True)

  # def call(self, inputs):
  #   return tf.matmul(inputs, self.w) + self.b

# linear_layer = Linear(4,2)


#In many cases, you may not know in advance the size of your inputs, and you would like to lazily create weights when that value becomes known, some time after instantiating the layer.
  def __init__(self, units=32):
    super(Linear, self).__init__()
    self.units = units

  def build(self, input_shape):
    self.w = self.add_weight(shape=(input_shape[-1], self.units),
                             initializer='random_normal',
                             trainable=True)
    self.b = self.add_weight(shape=(self.units,),
                             initializer='random_normal',
                             trainable=True)

  def call(self, inputs):
    return tf.matmul(inputs, self.w) + self.b

linear_layer = Linear(4)
print(linear_layer(tf.ones((2, 2))))
print(linear_layer.w)
print(linear_layer.b)
print('weights:', len(linear_layer.weights))
print('non-trainable weights:', len(linear_layer.non_trainable_weights))

# It's not included in the trainable weights:
print('trainable_weights:', linear_layer.trainable_weights)

'''
** Layer output **
tf.Tensor(
[[ 0.00501864  0.01122843  0.04163956 -0.01085223]
 [ 0.00501864  0.01122843  0.04163956 -0.01085223]], shape=(2, 4), dtype=float32)

** Weight matrix **
<tf.Variable 'linear/Variable:0' shape=(2, 4) dtype=float32, numpy=
array([[ 0.05603973, -0.04955313, -0.00082266,  0.04320528],
       [-0.00492885, -0.01754555,  0.03675178, -0.04522285]],
      dtype=float32)>

** Biaa Vector **
<tf.Variable 'linear/Variable:0' shape=(4,) dtype=float32, numpy=array([-0.04609224,  0.07832711,  0.00571045, -0.00883466], dtype=float32)>

** 2 parameters are trained weights and bias since for both trainable=True **
weights: 2
non-trainable weights: 0

** Weight and Bias values combined **
trainable_weights: [<tf.Variable 'linear/Variable:0' shape=(2, 4) dtype=float32, numpy=
array([[ 0.05603973, -0.04955313, -0.00082266,  0.04320528],
       [-0.00492885, -0.01754555,  0.03675178, -0.04522285]],
      dtype=float32)>, <tf.Variable 'linear/Variable:0' shape=(4,) dtype=float32, numpy=array([-0.04609224,  0.07832711,  0.00571045, -0.00883466], dtype=float32)>]
'''