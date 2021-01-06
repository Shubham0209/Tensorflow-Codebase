'''
You will build a Neural Machine Translation (NMT) model to translate human readable dates ("25th of June, 2009") into machine readable dates ("2009-06-25"). You will do this using an attention model, one of the most sophisticated sequence to sequence models.
'''
from tensorflow.keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from tensorflow.keras.layers import RepeatVector, Dense, Activation, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model, Model
import tensorflow.keras.backend as K
import numpy as np

from faker import Faker
import random
from tqdm import tqdm
from babel.dates import format_date
from nmt_utils import *
import matplotlib.pyplot as plt


m = 10000
dataset, human_vocab, machine_vocab, inv_machine_vocab = load_dataset(m)
'''
dataset: a list of tuples of (human readable date, machine readable date)
human_vocab: a python dictionary mapping all characters used in the human readable dates to an integer-valued index
machine_vocab: a python dictionary mapping all characters used in machine readable dates to an integer-valued index. These indices are not necessarily consistent with human_vocab.
inv_machine_vocab: the inverse dictionary of machine_vocab, mapping from indices back to characters
'''

print(dataset[:10]) #[('22 may 1999', '1999-05-22'), ('07.02.95', '1995-02-07'),.........,('sunday september 5 1971', '1971-09-05')]

Tx = 30 #which we assume is the maximum length of the human readable date; For longer input, we would have to truncate it.
Ty = 10 #YYYY-MM-DD" is 10 characters long.
X, Y, Xoh, Yoh = preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty) #preprocess the data and map the raw text data into the index values

'''
X: a processed version of the human readable dates in the training set, where each character is replaced by an index mapped to the character via human_vocab. Each date is further padded to T_x values with a special character (< pad >). X.shape = (m, Tx)
Y: a processed version of the machine readable dates in the training set, where each character is replaced by the index it is mapped to in machine_vocab. You should have Y.shape = (m, Ty).
Xoh: one-hot version of X, the "1" entry's index is mapped to the character thanks to human_vocab. Xoh.shape = (m, Tx, len(human_vocab))
Yoh: one-hot version of Y, the "1" entry's index is mapped to the character thanks to machine_vocab. Yoh.shape = (m, Tx, len(machine_vocab)). Here, len(machine_vocab) = 11 since there are 11 characters ('-' as well as 0-9).
'''
print("X.shape:", X.shape) #(10000, 30)
print("Y.shape:", Y.shape) #(10000, 10)
print("Xoh.shape:", Xoh.shape) # (10000, 30, 37)
print("Yoh.shape:", Yoh.shape) #(10000, 10, 11)


index = 0
print("Source date:", dataset[index][0]) #22 may 1999
print("Target date:", dataset[index][1]) #1999-05-22
print()
print("Source after preprocessing (indices):", X[index]) 
#[ 5  5  0 24 13 34  0  4 12 12 12 36 36 36  36 36 36 36 36 36 36 36 36 36 36 36 36 36 36 36]
#[ 2  2    m   a  y     1  9  9  4  <>  <>   <>  <>  <>  <>  <>  <>  <>  <>  <>  <>  <>  <>  <>]
print("Target after preprocessing (indices):", Y[index]) 
#[ 2 10 10 10  0  1  6  0  3  3]
#[ 1  9  9  9     0  5     2  2]
print()
print("Source after preprocessing (one-hot):", Xoh[index])
print("Target after preprocessing (one-hot):", Yoh[index])


# Defined shared layers as global variables
repeator = RepeatVector(Tx)
concatenator = Concatenate(axis=-1)
densor1 = Dense(10, activation = "tanh")
densor2 = Dense(1, activation = "relu")
activator = Activation(softmax, name='attention_weights') # We are using a custom softmax(axis = 1) loaded in this notebook
dotor = Dot(axes = 1)

def one_step_attention(a, s_prev):
    """
    Performs one step of attention: Outputs a context vector computed as a dot product of the attention weights
    "alphas" and the hidden states "a" of the Bi-LSTM.
    
    Arguments:
    a -- hidden state output of the Bi-LSTM, numpy-array of shape (m, Tx, 2*n_a)
    s_prev -- previous hidden state of the (post-attention) LSTM, numpy-array of shape (m, n_s)
    
    Returns:
    context -- context vector, input of the next (post-attetion) LSTM cell
    """
    
    ### START CODE HERE ###
    # Use repeator to repeat s_prev to be of shape (m, Tx, n_s) so that you can concatenate it with all hidden states "a" (≈ 1 line)
    s_prev = repeator(s_prev)
    # Use concatenator to concatenate a and s_prev on the last axis (≈ 1 line)
    concat = concatenator([a, s_prev])
    # Use densor1 to propagate concat through a small fully-connected neural network to compute the "intermediate energies" variable e. (≈1 lines)
    e = densor1(concat)
    # Use densor2 to propagate e through a small fully-connected neural network to compute the "energies" variable energies. (≈1 lines)
    energies = densor2(e)
    # Use activator and e to compute the attention weights "alphas" (≈ 1 line)
    alphas = activator(e)
    # Use dotor together with "alphas" and "a" to compute the context vector to be given to the next (post-attention) LSTM-cell (≈ 1 line)
    context = dotor([alphas, a])
    ### END CODE HERE ###
    
    return context

n_a = 64 # number of units for the pre-attention, bi-directional LSTM's hidden state 'a'
n_s = 128 ## number of units for the post-attention LSTM's hidden state "s"
post_activation_LSTM_cell = LSTM(n_s, return_state = True)
output_layer = Dense(len(machine_vocab), activation=softmax)

def model(Tx, Ty, n_a, n_s, human_vocab_size, machine_vocab_size):
    """
    Arguments:
    Tx -- length of the input sequence
    Ty -- length of the output sequence
    n_a -- hidden state size of the Bi-LSTM
    n_s -- hidden state size of the post-attention LSTM
    human_vocab_size -- size of the python dictionary "human_vocab"
    machine_vocab_size -- size of the python dictionary "machine_vocab"

    Returns:
    model -- Keras model instance
    """
    
    # Define the inputs of your model with a shape (Tx,)
    # Define s0 and c0, initial hidden state for the decoder LSTM of shape (n_s,)
    X = Input(shape=(Tx, human_vocab_size))
    s0 = Input(shape=(n_s,), name='s0')
    c0 = Input(shape=(n_s,), name='c0')
    s = s0
    c = c0
    
    # Initialize empty list of outputs
    outputs = []
    
    ### START CODE HERE ###
    
    # Step 1: Define your pre-attention Bi-LSTM. Remember to use return_sequences=True. (≈ 1 line)
    a = Bidirectional(LSTM(n_a, return_sequences=True))(X)
    
    # Step 2: Iterate for Ty steps
    for t in range(Ty):
    
        # Step 2.A: Perform one step of the attention mechanism to get back the context vector at step t (≈ 1 line)
        context = one_step_attention(a, s)
        
        # Step 2.B: Apply the post-attention LSTM cell to the "context" vector.
        # Don't forget to pass: initial_state = [hidden state, cell state] (≈ 1 line)
        s, _, c = post_activation_LSTM_cell(context, initial_state = [s, c])
        
        # Step 2.C: Apply Dense layer to the hidden state output of the post-attention LSTM (≈ 1 line)
        out = output_layer(s)
        
        # Step 2.D: Append "out" to the "outputs" list (≈ 1 line)
        outputs.append(out)
    
    # Step 3: Create model instance taking three inputs and returning the list of outputs. (≈ 1 line)
    model = Model(inputs = [X, s0, c0], outputs = outputs)
    
    ### END CODE HERE ###
    
    return model
model = model(Tx, Ty, n_a, n_s, len(human_vocab), len(machine_vocab))
out = model.compile(optimizer=Adam(lr=0.005, beta_1=0.9, beta_2=0.999, decay=0.01),
                    metrics=['accuracy'],
                    loss='categorical_crossentropy')

s0 = np.zeros((m, n_s))
c0 = np.zeros((m, n_s))
outputs = list(Yoh.swapaxes(0,1))

model.fit([Xoh, s0, c0], outputs, epochs=10, batch_size=100)

EXAMPLES = ['3 May 1979', '5 April 09', '21th of August 2016', 'Tue 10 Jul 2007', 'Saturday May 9 2018', 'March 3 2001', 'March 3rd 2001', '1 March 2001']
for example in EXAMPLES:
    
    source = string_to_int(example, Tx, human_vocab)
    source = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), source))).swapaxes(0,1)
    prediction = model.predict([source, s0, c0])
    prediction = np.argmax(prediction, axis = -1)
    output = [inv_machine_vocab[int(i)] for i in prediction]
    
    print("source:", example)
    print("output:", ''.join(output),"\n")