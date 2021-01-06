import os
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

base_dir = 'cats_and_dogs_filtered'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# Directory with our training cat/dog pictures
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
# Directory with our validation cat/dog pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

train_cat_fnames = os.listdir( train_cats_dir )
train_dog_fnames = os.listdir( train_dogs_dir )

# print(train_cat_fnames[:10])
# print(train_dog_fnames[:10])

# print('total training cat images :', len(os.listdir(      train_cats_dir ) ))
# print('total training dog images :', len(os.listdir(      train_dogs_dir ) ))

# print('total validation cat images :', len(os.listdir( validation_cats_dir ) ))
# print('total validation dog images :', len(os.listdir( validation_dogs_dir ) ))

# All images will be rescaled by 1./255.
train_datagen = ImageDataGenerator( rescale = 1.0/255. )
# code for image augmentation
# train_datagen = ImageDataGenerator(
#        rescale = 1.0/255.,
# 	  rotation_range=40, # rotate the image by random amount between 0 to 40degrees
#       width_shift_range=0.2, # shift the image within its frame by 20%
#       height_shift_range=0.2,
#       shear_range=0.2,
#       zoom_range=0.2, #zoom in the image by 20%
#       horizontal_flip=True, # flip the image along vertival axis
#       fill_mode='nearest') # fill in any pixels which are lost using nearest neighjbors
test_datagen  = ImageDataGenerator( rescale = 1.0/255. )

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=20,
                                                    class_mode='binary',
                                                    shuffle = True,
                                                    target_size=(150, 150))     
validation_generator =  test_datagen.flow_from_directory(validation_dir,
                                                         batch_size=20,
                                                         class_mode  = 'binary',
                                                         target_size = (150, 150))

# Visualization
# The next function returns a batch from the dataset. The return value of next function is in form of (x_train, y_train)
# sample_training_images, _ = next(train_data_gen)
# plt.imshow(sample_training_images[0])

model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    # tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2), 
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'), 
    tf.keras.layers.MaxPooling2D(2,2),
    # tf.keras.layers.Dropout(0.2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(), 
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'), 
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('cats') and 1 for the other ('dogs')
    tf.keras.layers.Dense(1, activation='sigmoid')  
])

model.compile(optimizer=RMSprop(lr=0.001),
              loss='binary_crossentropy',
              metrics = ['accuracy']) # generally only loss printed but here accuracy along with loss printed

history = model.fit(train_generator,
                              validation_data=validation_generator,
                              steps_per_epoch=100, #since 2000 total train images in batch of 20 gives 100 steps
                              epochs=15,
                              validation_steps=50, #since 1000 total test images in batch of 20 gives 50 steps
                              verbose=2) # to print loss and accuracy of both test and train data
# You'll see 4 values per epoch -- Loss, Accuracy, Validation Loss and Validation Accuracy.

acc      = history.history[     'accuracy' ]
val_acc  = history.history[ 'val_accuracy' ]
loss     = history.history[    'loss' ]
val_loss = history.history['val_loss' ]
epochs   = range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.plot  ( epochs,     acc )
plt.plot  ( epochs, val_acc )
plt.title ('Training and validation accuracy')
plt.figure()