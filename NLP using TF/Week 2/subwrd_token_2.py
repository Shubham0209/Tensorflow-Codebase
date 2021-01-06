#Here text is case sensitive so Dat and dat are mapped to diffrent word index. Moreover punctuation is maintained

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import io


embedding_dim = 64

training_sentences = []
training_labels = []
imdb, info = tfds.load("imdb_reviews/subwords8k", with_info=True, as_supervised=True) #tokenized and encoded imdb dataset with vocab size of 8k i.e tokenizer.vocab_size = 8185
train_data, test_data = imdb['train'], imdb['test']
tokenizer = info.features['text'].encoder # The dataset info includes the text encode


print(len(tokenizer.subwords)) #7928
print(tokenizer.subwords) #['the_', ', ', '. ', 'a_', 'and_', 'of_', 'to_', 's_', 'is_', 'br', 'in_', 'I_', 'that_', 'this_', 'it_', ' /><',.........]

BUFFER_SIZE = 10000
BATCH_SIZE = 64

train_data = train_data.shuffle(BUFFER_SIZE)
train_data = train_data.padded_batch(BATCH_SIZE, padded_shapes=([None],[]))
test_data = test_data.padded_batch(BATCH_SIZE,padded_shapes=([None],[]))


sample = 'Data Science is a very cool, and amazing thing to learn'
token_string = tokenizer.encode(sample) #[878, 1848, 2675, 2975, 9, 4, 67, 2724, 2, 5, 1006, 233, 7, 5635]

print(token_string)

orig_string = tokenizer.decode(token_string)
print(orig_string)

print(tokenizer.vocab_size) #8185

for t in token_string:
	print('{} --> {}'.format(t, tokenizer.decode([t])))
'''
878 --> Da
1848 --> ta
2675 --> Sci
2975 --> ence
9 --> is
4 --> a
67 --> very
2724 --> cool
2 --> ,
5 --> and
1006 --> amazing
233 --> thing
7 --> to
5635 --> learn
'''
model = tf.keras.Sequential([
	
    tf.keras.layers.Embedding(input_dim = tokenizer.vocab_size, output_dim = embedding_dim), #The results of the embedding will be a 2D vector with one embedding for each word in the input sequence of words (input sentence)
    tf.keras.layers.GlobalAveragePooling1D(), #Often in natural language processing, a different layer type than a Flatten is used, and this is a GlobalAveragePooling1D. The reason for this is the size of the output vector being fed into the dense
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

num_epochs = 10
model.fit(train_data, epochs=num_epochs, validation_data=test_data)

