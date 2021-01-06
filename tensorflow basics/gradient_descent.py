import tensorflow as tf
#By default GradientTape will automatically watch any trainable variables that are accessed inside the context. 
#If you want fine grained control over which variables are watched you can disable automatic tracking by passing watch_accessed_variables=False to the tape constructor:

#Code -1
# with tf.GradientTape(watch_accessed_variables=False) as tape:
#   tape.watch(variable_a)
#   y = variable_a ** 2  # Gradients will be available for `variable_a`.
#   z = variable_b ** 3  # No gradients will be available since `variable_b` is
#                        # not being watched.

#Code-2
# x = tf.constant(3.0)
# with tf.GradientTape() as g:
#   g.watch(x)
#   with tf.GradientTape() as gg:
#     gg.watch(x)
#     y = x * x
#   dy_dx = gg.gradient(y, x)     # Will compute to 6.0
# d2y_dx2 = g.gradient(dy_dx, x)  # Will compute to 2.0
# print(dy_dx)
# print(d2y_dx2)

#Code-3
x = tf.constant(3.0)
with tf.GradientTape(persistent=True) as g:
  g.watch(x)
  y = x * x
  z = y * y
dz_dx = g.gradient(z, x)  # 108.0 (4*x^3 at x = 3)
dy_dx = g.gradient(y, x)  # 6.0
del g  # Drop the reference to the tape