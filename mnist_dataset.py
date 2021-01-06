import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# class myCallback(tf.keras.callbacks.Callback):
#   def on_epoch_end(self, epoch, logs={}):
#     if(logs.get('loss')<0.4):
#       print("\nReached 60% accuracy so cancelling training!")
#       self.model.stop_training = True

# callbacks = myCallback()
# mnist = tf.keras.datasets.fashion_mnist
# (training_images, training_labels), (test_images, test_labels) = mnist.load_data()
# training_images=training_images/255.0
# test_images=test_images/255.0
# model = tf.keras.models.Sequential([
#   tf.keras.layers.Flatten(),
#   tf.keras.layers.Dense(512, activation=tf.nn.relu),
#   tf.keras.layers.Dense(10, activation=tf.nn.softmax)
# ])
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
# model.fit(training_images, training_labels, epochs=5, callbacks=[callbacks])




mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

np.set_printoptions(linewidth=200)
plt.imshow(training_images[0])
plt.colorbar()
plt.grid(False)
print(training_labels[0])
print(training_images[0])

training_images  = training_images / 255.0
test_images = test_images / 255.0

# Prining random images

# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)	
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()



model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)), #no need to define i/p size as keras can figure it out automatically
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu), 
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

model.compile(optimizer = tf.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=5)
                              
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose =2 )
print('\nTest accuracy:', test_acc)

predictions = model.predict(test_images)
print(np.argmax(predictions[0]))
print(test_labels[0])