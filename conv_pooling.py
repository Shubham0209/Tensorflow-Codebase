import tensorflow as tf
print(tf.__version__)
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
training_images=training_images.reshape(60000, 28, 28, 1) #  first convolution expects a single tensor containing everything, so instead of 60,000 28x28x1 items in a list, we have a single 4D list that is 60,000x28x28x1
training_images=training_images / 255.0
test_images = test_images.reshape(10000, 28, 28, 1)
test_images=test_images/255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),#generate 64 filters of size 3*3; i/p is 28*28 with 1 depth since grayscale
  tf.keras.layers.MaxPooling2D(2, 2),# poolog layer i.e take 2*2 pixels at a time ans take max of it{MaxPooling}
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),# another convlutional layer
  tf.keras.layers.MaxPooling2D(2,2), # another pooling layer
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 26, 26, 64)        640 [(3*3*1*64)+64]
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 13, 13, 64)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 11, 11, 64)        36928[(3*3*64*64)+64]
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 5, 5, 64)          0
_________________________________________________________________
flatten (Flatten)            (None, 1600)              0
_________________________________________________________________
dense (Dense)                (None, 128)               204928[(1600*128)+128]
_________________________________________________________________
dense_1 (Dense)              (None, 10)                1290[(128*10)+10]
=================================================================
Total params: 243,786
Trainable params: 243,786
Non-trainable params: 0
_________________________________________________________________
'''
model.fit(training_images, training_labels, epochs=1)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(test_acc)

