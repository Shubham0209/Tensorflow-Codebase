import csv
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = ['I love my cat',
			 'I, love my dog!',
			 'Do you think my dog is amazing?'
			 ]


tokenizer = Tokenizer(num_words = 100, oov_token ="<OOV>") # in your body of texts that it's tokenizing, it will take the 100 most common words and specified that I want the token oov for out of vocabulary to be used for words that aren't in the word index.
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print(len(word_index)) # {'<OOV>': 1, 'my': 2, 'i': 3, 'love': 4, 'dog': 5, 'cat': 6, 'do': 7, 'you': 8, 'think': 9, 'is': 10, 'amazing': 11}
print((word_index))
sequences = tokenizer.texts_to_sequences(sentences) 
print(sequences) # [[3, 4, 2, 6], [3, 4, 2, 5], [7, 8, 9, 2, 5, 10, 11]]
padded = pad_sequences(sequences, padding ='post') 
print(padded) 
'''
[[ 3  4  2  6  0  0  0] # the matrix width was the same as the longest sentence. But you can override that with the 'maxlen' parameter in pad_sequences func
 [ 3  4  2  5  0  0  0]
 [ 7  8  9  2  5 10 11]]
'''

test_data = ['i really love my dog', 'my dog loves manatee']
test_seq = tokenizer.texts_to_sequences(test_data)
print(test_seq) # [[3, 1, 4, 2, 5], [2, 5, 1, 1]]
padded_test = pad_sequences(test_seq, padding ='post') 
print(padded_test) 
'''
[[3 1 4 2 5]
 [2 5 1 1 0]]
 '''