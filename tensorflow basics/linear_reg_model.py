import tensorflow as tf
import  matplotlib.pyplot as plt

TRUE_W = 3.0
TRUE_b = 2.0
NUM_EXAMPLES = 1000

inputs  = tf.random.normal(shape=[NUM_EXAMPLES])
noise   = tf.random.normal(shape=[NUM_EXAMPLES])
actual_outputs = inputs * TRUE_W + TRUE_b + noise

class Model(object):
  
  def __init__(self):
    # Initialize the weights to `5.0` and the bias to `0.0`
    # In practice, these should be initialized to random values (for example, with `tf.random.normal`)
    self.W = tf.Variable(5.0)
    self.b = tf.Variable(0.0)

  def __call__(self, x):
    return self.W * x + self.b

def loss(target_y, predicted_y):
  return tf.reduce_mean(tf.square(target_y - predicted_y))

def train(model, inputs, outputs, learning_rate):
  with tf.GradientTape() as t:
    current_loss = loss(outputs, model(inputs))
  dW, db = t.gradient(current_loss, [model.W, model.b])
  #Update w and b based on gradient
  model.W.assign_sub(learning_rate * dW)
  model.b.assign_sub(learning_rate * db)

model = Model()

# Collect the history of W-values and b-values to plot later
Ws, bs = [], []
epochs = range(10)
for epoch in epochs:
  Ws.append(model.W.numpy())
  bs.append(model.b.numpy())
  current_loss = loss(actual_outputs, model(inputs))

  train(model, inputs, actual_outputs, learning_rate=0.1)
  print('Epoch %2d: W=%1.2f b=%1.2f, loss=%2.5f' %
        (epoch, Ws[-1], bs[-1], current_loss))

# Let's plot it all
plt.plot(epochs, Ws, 'r',
         epochs, bs, 'b')
plt.plot([TRUE_W] * len(epochs), 'r--',
         [TRUE_b] * len(epochs), 'b--')
plt.legend(['W', 'b', 'True W', 'True b'])
plt.show()

# Epoch  0: W=5.00 b=0.00, loss=9.00746
# Epoch  1: W=4.58 b=0.39, loss=6.07767
# Epoch  2: W=4.25 b=0.70, loss=4.21756
# Epoch  3: W=3.99 b=0.95, loss=3.03633
# Epoch  4: W=3.78 b=1.15, loss=2.28608
# Epoch  5: W=3.62 b=1.32, loss=1.80945
# Epoch  6: W=3.49 b=1.45, loss=1.50660
# Epoch  7: W=3.39 b=1.55, loss=1.31413
# Epoch  8: W=3.30 b=1.63, loss=1.19179
# Epoch  9: W=3.24 b=1.70, loss=1.11401