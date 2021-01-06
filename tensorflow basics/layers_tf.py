import tensorflow as tf

# In the tf.keras.layers package, layers are objects. To construct a layer,
# simply construct the object. Most layers take as a first argument the number
# of output dimensions / channels.
layer = tf.keras.layers.Dense(100)
# The number of input dimensions is often unnecessary, as it can be inferred
# the first time the layer is used, but it can be provided if you want to
# specify it manually, which is useful in some complex models.
layer = tf.keras.layers.Dense(10, input_shape=(None, 5))

# To use a layer, simply call it.
print(layer(tf.zeros([10, 5])))
'''
tf.Tensor(
[[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]], shape=(10, 10), dtype=float32)
'''
# Layers have many useful methods. For example, you can inspect all variables
# in a layer using `layer.variables` and trainable variables using
# `layer.trainable_variables`. In this case a fully-connected layer
# will have variables for weights and biases.
print(layer.variables)
'''
[
** for kernel/weight matrix ** -> can also be accessed using layer.kernel
** these are randomly initilaed weights according to Xavier distribution (deafult)**
<tf.Variable 'dense_1/kernel:0' shape=(5, 10) dtype=float32, numpy=
array([[-0.3917894 , -0.03007638, -0.10666072, -0.38480562,  0.22483945,
         0.2906363 ,  0.2021609 , -0.09012514,  0.26172036, -0.30604452],
       [ 0.5540382 , -0.5287375 , -0.47426593,  0.25994468,  0.17369658,
         0.5005253 , -0.08237171, -0.6314669 , -0.6020418 , -0.59982145],
       [-0.33839768, -0.10360605,  0.4495986 ,  0.04161346,  0.5868183 ,
        -0.31909803, -0.44844905,  0.02731365, -0.39546144, -0.5518584 ],
       [ 0.21691966, -0.49045467,  0.4199087 ,  0.6165815 , -0.5667673 ,
         0.15355718, -0.47339064, -0.01113188, -0.47745603, -0.00933069],
       [-0.14388934,  0.3212207 , -0.5475133 ,  0.41148108,  0.18656284,
         0.57632226,  0.19484478,  0.11316228,  0.1051293 ,  0.25014365]],
      dtype=float32)>, 

** for bias** -> can also be accessed using layer.bias
<tf.Variable 'dense_1/bias:0' shape=(10,) dtype=float32, numpy=array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=float32)>
]
'''
