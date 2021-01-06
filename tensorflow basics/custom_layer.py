import tensorflow as tf
'''
__init__ : where you can do all input-independent initialization
build: where you know the shapes of the input tensors and can do the rest of the initialization
call: where you do the forward computation
The advantage of creating them in build is that it enables late variable creation based on the shape of the inputs the layer will operate on. 
On the other hand, creating variables in __init__ would mean that shapes required to create the variables will need to be explicitly specified.
'''
class MyDenseLayer(tf.keras.layers.Layer):
  def __init__(self, num_outputs):
    super(MyDenseLayer, self).__init__()
    self.num_outputs = num_outputs

  def build(self, input_shape):
    self.kernel = self.add_weight("kernel",
                                  shape=[int(input_shape[-1]),
                                         self.num_outputs])

  def call(self, input):
    return tf.matmul(input, self.kernel)
## The __call__ method of your layer will automatically run build the first time it is called
layer = MyDenseLayer(10)
print(layer(tf.zeros([7, 5]))) # Calling the layer `.builds` it.
'''
tf.Tensor(
[[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]], shape=(7, 10), dtype=float32)
 '''
print(layer.variables)
'''
[<tf.Variable 'my_dense_layer/kernel:0' shape=(5, 10) dtype=float32, numpy=
array([[-0.44193798,  0.03905165,  0.04860175,  0.17380154,  0.61841327,
         0.24468601, -0.22788855, -0.49311218, -0.18143809,  0.5089671 ],
       [-0.62658936,  0.5966634 , -0.31927943, -0.5823326 , -0.20540619,
        -0.17128953,  0.00555354, -0.5033939 ,  0.42664188, -0.03926849],
       [-0.11438775, -0.08078319, -0.085527  ,  0.12992203, -0.14082938,
         0.0564363 ,  0.62760025,  0.05861324,  0.23688537, -0.59837115],
       [-0.49668074, -0.33774596, -0.34752932, -0.29713243, -0.3249921 ,
         0.6053243 , -0.5222039 , -0.5378549 ,  0.43061322,  0.5299671 ],
       [-0.47661102, -0.07770771, -0.55105895,  0.01929271,  0.5278507 ,
         0.5874421 ,  0.3264674 ,  0.30255282, -0.35374802,  0.2447077 ]],
      dtype=float32)>]

** No bias printed this time **