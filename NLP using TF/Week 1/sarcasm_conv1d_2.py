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
print(len(word_index)) #25637 ALthough here length shown as 25k but only top 10k will be used for encoding

training_seq = tokenizer.texts_to_sequences(training_sentence)
training_padded = pad_sequences(training_seq,maxlen=max_length, truncating=trunc_type, padding = padding_typ)
print(training_padded[0])
print(training_padded.shape)

testing_seq = tokenizer.texts_to_sequences(testing_sentence)
testing_padded = pad_sequences(testing_seq,maxlen=max_length, truncating=trunc_type, padding = padding_typ)
print(testing_padded[0])
print(testing_padded.shape)

model = tf.keras.Sequential([
	tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Conv1D(128, 5, activation='relu'), #we have 128 filters each for 5 words. This line can be commented as well
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

num_epochs = 30
model.fit(training_padded, training_label, epochs=num_epochs, validation_data=(testing_padded, testing_label), verbose = 2)

'''
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_1 (Embedding)      (None, 120, 16)           16000     
_________________________________________________________________
conv1d_1 (Conv1D)            (None, 116, 128)          10368     #the size of the input was 120 words, and a filter that is 5 words long will shave off 2 words from the front and back, leaving us with 116. 
_________________________________________________________________
global_max_pooling1d_1 (Glob (None, 128)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 24)                3096      
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 25        
=================================================================
Total params: 29,489
Trainable params: 29,489
Non-trainable params: 0
________________________