import tensorflow as tf

a = tf.constant([[1, 3], [2, 0], [0, 1]])
# Sum elements from different parts of the array.
# ... With no second argument, everything is summed.
#     Second argument indicates what axis to sum.
b = tf.reduce_sum(a)
c = tf.reduce_sum(a, 0)
d = tf.reduce_sum(a, 1)

# Just for looping.
tensors = [a, b, c, d]

# Loop over our tensors and run a Session on the graph.
for tensor in tensors:
 	# result = tf.Session().run(tensor)
    print(tensor)