import tensorflow as tf
import numpy as np
import time

print(tf.add(1, 2))
print(tf.add([1, 2], [3, 4])) # it is vector as shown
# |1|  |3|
# |2|  |4|
print(tf.square(5))
print(tf.reduce_sum([1, 2, 3]))

# Operator overloading is also supported
print(tf.square(2) + tf.square(3))

'''
tf.Tensor(3, shape=(), dtype=int32)
tf.Tensor([4 6], shape=(2,), dtype=int32)
tf.Tensor(25, shape=(), dtype=int32)
tf.Tensor(6, shape=(), dtype=int32)
tf.Tensor(13, shape=(), dtype=int32)
'''

x = tf.multiply([[1]], [[2, 3]]) # it is matrix as shown below << tf.matmul() also works>>
# |1 |  |2 3|
print(x)
print(x.shape)
print(x.dtype)

'''
tf.Tensor([[2 3]], shape=(1, 2), dtype=int32)
(1, 2)
<dtype: 'int32'>
'''

ndarray = np.ones([3, 3])

print("TensorFlow operations convert numpy arrays to Tensors automatically")
tensor = tf.multiply(ndarray, 42)
print(tensor)


print("And NumPy operations convert Tensors to numpy arrays automatically")
print(np.add(tensor, 1))

print("The .numpy() method explicitly converts a Tensor to a numpy array")
print(tensor.numpy())

'''
TensorFlow operations convert numpy arrays to Tensors automatically
tf.Tensor(
[[42. 42. 42.]
 [42. 42. 42.]
 [42. 42. 42.]], shape=(3, 3), dtype=float64)
And NumPy operations convert Tensors to numpy arrays automatically
[[43. 43. 43.]
 [43. 43. 43.]
 [43. 43. 43.]]
The .numpy() method explicitly converts a Tensor to a numpy array
[[42. 42. 42.]
 [42. 42. 42.]
 [42. 42. 42.]]
 '''

x = tf.random.uniform([3, 3])

print("Is there a GPU available: "),
print(tf.config.experimental.list_physical_devices("GPU"))

print("Is the Tensor on GPU #0:  "),
print(x.device.endswith('GPU:0')) # checks if the tensor x is placed on GPU 0

'''
Is there a GPU available:
[]
Is the Tensor on GPU #0:
False
'''

def time_matmul(x):
  start = time.time()
  for loop in range(10):
    tf.matmul(x, x)

  result = time.time()-start

  print("10 loops: {:0.2f}ms".format(1000*result))

if tf.config.experimental.list_physical_devices("GPU"):
  print("On GPU:")
  with tf.device("GPU:0"): # Or GPU:1 for the 2nd GPU, GPU:2 for the 3rd etc.
    x = tf.random.uniform([1000, 1000])
    assert x.device.endswith("GPU:0") #explicitly assign tensor to GPU0
    time_matmul(x)

v = tf.Variable(2.0)
w = v + 1  # w is a tf.Tensor which is computed based on the value of v.
           # Any time a variable is used in an expression it gets automatically
           # converted to a tf.Tensor representing its value.
print(w)
print(v)
print(v.assign_add(1))# assign by adding a value to a variable,... to just assign -> v.assign(1)
print(v.read_value()) 
'''
tf.Tensor(3.0, shape=(), dtype=float32)
<tf.Variable 'Variable:0' shape=() dtype=float32, numpy=2.0>
<tf.Variable 'UnreadVariable' shape=() dtype=float32, numpy=3.0>
tf.Tensor(3.0, shape=(), dtype=float32)
'''

'''
A Variable in TensorFlow is a Python object. 
As you build your layers, models, optimizers, and other related tools, you will likely want to get a list of all variables in a (say) model.
A common use case is implementing Layer subclasses. The Layer class recursively tracks variables set as instance attributes
'''
class MyLayer(tf.keras.layers.Layer):

  def __init__(self):
    super(MyLayer, self).__init__()
    self.my_var = tf.Variable(1.0)
    self.my_var_list = [tf.Variable(x) for x in range(10)]

class MyOtherLayer(tf.keras.layers.Layer):

  def __init__(self):
    super(MyOtherLayer, self).__init__()
    self.sublayer = MyLayer()
    self.my_other_var = tf.Variable(10.0)

m = MyOtherLayer()
print(len(m.variables))  # 12 (11 from MyLayer, plus my_other_var)