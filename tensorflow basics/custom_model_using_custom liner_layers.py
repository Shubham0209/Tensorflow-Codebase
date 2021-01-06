import tensorflow as tf
from custom_layer_with_biasV2 import Linear

class MLPBlock(tf.keras.layers.Layer):

  def __init__(self):
    super(MLPBlock, self).__init__()
    # Creates 3 intsnaces of linear class
    self.linear_1 = Linear(32)
    self.linear_2 = Linear(32)
    self.linear_3 = Linear(1)

  def call(self, inputs):
    x = self.linear_1(inputs) 
    x = tf.nn.relu(x)
    x = self.linear_2(x)
    x = tf.nn.relu(x)
    return self.linear_3(x)


mlp = MLPBlock()
y = mlp(tf.ones(shape=(3, 64)))  # The first call to the `mlp` will create the weights
print('weights:', len(mlp.weights))
print('trainable weights:', len(mlp.trainable_weights))
'''
tf.Tensor( --> from custom_layer_with_biasV2 code
[[ 0.06864086 -0.0451211   0.03909843 -0.0699331 ]
 [ 0.06864086 -0.0451211   0.03909843 -0.0699331 ]], shape=(2, 4), dtype=float32)
<tf.Variable 'linear/Variable:0' shape=(2, 4) dtype=float32, numpy=
array([[ 0.00085538, -0.0124227 ,  0.03137842, -0.01236949],
       [ 0.05244306,  0.00293   ,  0.06805115, -0.02822451]],
      dtype=float32)>
<tf.Variable 'linear/Variable:0' shape=(4,) dtype=float32, numpy=array([ 0.01534242, -0.0356284 , -0.06033114, -0.0293391 ], dtype=float32)>
weights: 6
trainable weights: 6
''