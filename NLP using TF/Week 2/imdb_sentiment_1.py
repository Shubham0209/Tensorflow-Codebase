#we will look at how we can learn a word embedding while fitting a neural network on a sentiment classification problem.

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import io

vocab_size = 10000
embedding_dim = 16
max_length = 120
trunc_type='post'
oov_tok = "<OOV>"

imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)
train_data, test_data = imdb['train'], imdb['test']

training_sentences = []
training_labels = []

testing_sentences = []
testing_labels = []

# str(s.tonumpy()) is needed in Python3 instead of just s.numpy()
#  The values for S and L are tensors, so by calling their NumPy method, I'll actually extract their value.
for s,l in train_data:
  training_sentences.append(str(s.numpy()))
  training_labels.append(l.numpy())
'''
s looks like
tf.Tensor(b"This was an absolutely terrible movie. Don't be lured in by Christopher Walken or Michael Ironside. 
Both are great actors, but this must simply be their worst role in history. Even their great acting could not redeem this movie's ridiculous storyline.
This movie is an early nineties US propaganda piece. The most pathetic scenes were those when the Columbian rebels were making their cases for revolutions. 
Maria Conchita Alonso appeared phony, and her pseudo-love affair with Walken was nothing but a pathetic emotional plug in a movie that was devoid of any real meaning. 
I am disappointed that there are movies like this, ruining actor's like Christopher Walken's good name. 
I could barely sit through it.", shape=(), dtype=string)

l looks like
tf.Tensor(0, shape=(), dtype=int64) -> 0 means negtive review
'''
for s,l in test_data:
  testing_sentences.append(str(s.numpy()))
  testing_labels.append(l.numpy())
  
training_labels_final = np.array(training_labels) #converts labels list to array
testing_labels_final = np.array(testing_labels)

tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index # size 86539 but we keep only top 10k hence the words which are not taken in word_index are reaplaced with oov
sequences = tokenizer.texts_to_sequences(training_sentences)
padded = pad_sequences(sequences,maxlen=max_length, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences,maxlen=max_length)

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text]) #dict.get(key, default=None) i.e 0 won't be there in index so replace it with ?

print(decode_review(padded[3]))
print(training_sentences[3])

'''
	Embedding requires that the input data be integer encoded, so that each word is represented by a unique integer -> this step is performed by tokenization.
    
    Input_dim: This is the size of the vocabulary in the text data. For example, if your data is integer encoded to values between 0-10, then the size of the vocabulary would be 11 words.
	output_dim: This is the size of the vector space in which words will be embedded. It defines the size of the output vectors from this layer for each word. For example, it could be 32 or 100 or even larger. Test different values for your problem.
	input_length: This is the length of input sequences, as you would define for any input layer of a Keras model. For example, if all of your input documents are comprised of 1000 words, this would be 1000.
    '''
model = tf.keras.Sequential([
	
    tf.keras.layers.Embedding(input_dim = vocab_size, output_dim = embedding_dim, input_length=max_length), #The results of the embedding will be a 2D vector with one embedding for each word in the input sequence of words (input sentence)
    tf.keras.layers.Flatten(), #Often in natural language processing, a different layer type than a Flatten is used, and this is a GlobalAveragePooling1D. The reason for this is the size of the output vector being fed into the dense
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding (Embedding)        (None, 120, 16)           160000    
_________________________________________________________________
flatten (Flatten)            (None, 1920)              0      # with  GlobalAveragePooling1D (None,16)   0  as it averages across the vector to flatten it out
_________________________________________________________________
dense (Dense)                (None, 6)                 11526   #with  GlobalAveragePooling1D (None,6)   102  
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 7         
=================================================================
Total params: 171,533
Trainable params: 171,533
Non-trainable params: 0
_________________________________________________________________
'''	
num_epochs = 10
model.fit(padded, training_labels_final, epochs=num_epochs, validation_data=(testing_padded, testing_labels_final))

# visualize the embeddings
e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape) # shape: (vocab_size, embedding_dim)

out_v = io.open('vecs.tsv', 'w', encoding='utf-8') #file to store the word of the vocabulary
out_m = io.open('meta.tsv', 'w', encoding='utf-8') #file to store the emdedding of that word
for word_num in range(1, vocab_size): # because dictionary or word_index starts from 1
  word = reverse_word_index[word_num]
  embeddings = weights[word_num]
  out_m.write(word + "\n")
  out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
out_v.close()
out_m.close()