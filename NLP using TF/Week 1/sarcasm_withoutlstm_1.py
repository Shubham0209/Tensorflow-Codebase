import json
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


vocab_size = 10000
embedding_dim = 16
max_length = 120
trunc_type='post'
oov_tok = "<OOV>"
padding_typ = 'post'
training_size = 20000

sentences = []
labels = []
with open("sarcasm.json", 'r') as f:
    dreader = json.load(f)
    
for row in dreader:
	labels.append(row['is_sarcastic'])
	sentences.append(row['headline'])


print(len(sentences)) #26709
print(sentences[0]) 
'''
former versace store clerk sues over secret 'black code' for minority shoppers
'''
training_sentence = sentences[0:training_size]
testing_sentence = sentences[training_size:]
training_label = np.array(labels[0:training_size])
testing_label = np.array(labels[training_size:])


tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentence)
word_index = tokenizer.word_index
print(len(word_index)) #25637

training_seq = tokenizer.texts_to_sequences(training_sentence)
training_padded = pad_sequences(training_seq,maxlen=max_length, truncating=trunc_type, padding = padding_typ)
print(training_padded[0]) #[ 328    1  799 3405 2404   47  389 2214    1    6 2614 8863    0    0     0    0    0    0    0    0    0    0    0    0    0    0    0    0     0    0    0    0]
print(training_padded.shape) #(2000, 32)

testing_seq = tokenizer.texts_to_sequences(testing_sentence)
testing_padded = pad_sequences(testing_seq,maxlen=max_length, truncating=trunc_type, padding = padding_typ)
print(testing_padded[0])
print(testing_padded.shape) #(6709, 32)

model = tf.keras.Sequential([
	
    tf.keras.layers.Embedding(input_dim = vocab_size, output_dim = embedding_dim, input_length=max_length), #The results of the embedding will be a 2D vector with one embedding for each word in the input sequence of words (input sentence)
    tf.keras.layers.GlobalAveragePooling1D(), #Often in natural language processing, a different layer type than a Flatten is used, and this is a GlobalAveragePooling1D. The reason for this is the size of the output vector being fed into the dense
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

num_epochs = 30
model.fit(training_padded, training_label, epochs=num_epochs, validation_data=(testing_padded, testing_label), verbose = 2)
'''
Model: "sequential_2"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_2 (Embedding)      (None, 120, 16)           16000     
_________________________________________________________________
global_max_pooling1d_2 (Glob (None, 16)                0         
_________________________________________________________________
dense_4 (Dense)              (None, 24)                408       
_________________________________________________________________
dense_5 (Dense)              (None, 1)                 25        
=================================================================
Total params: 16,433
Trainable params: 16,433
Non-trainable params: 0