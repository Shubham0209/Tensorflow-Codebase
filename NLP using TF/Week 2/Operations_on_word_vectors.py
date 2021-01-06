import numpy as np
from w2v_utils import *
words, word_to_vec_map = read_glove_vecs('glove.6B.100d.txt')

def cosine_similarity(u, v):
    """
    Cosine similarity reflects the degree of similariy between u and v
        
    Arguments:
        u -- a word vector of shape (n,)          
        v -- a word vector of shape (n,)

    Returns:
        cosine_similarity -- the cosine similarity between u and v defined by the formula above.
    """
    
    distance = 0.0
    
    ### START CODE HERE ###
    # Compute the dot product between u and v (≈1 line)
    dot = np.dot(u, v)
    # Compute the L2 norm of u (≈1 line)
    norm_u = np.sqrt(np.sum(u * u))
    
    # Compute the L2 norm of v (≈1 line)
    norm_v = np.sqrt(np.sum(v * v))
    # Compute the cosine similarity defined by formula (1) (≈1 line)
    cosine_similarity = dot / (norm_u * norm_v)
    ### END CODE HERE ###
    
    return cosine_similarity


father = word_to_vec_map["father"]
mother = word_to_vec_map["mother"]
ball = word_to_vec_map["ball"]
crocodile = word_to_vec_map["crocodile"]
france = word_to_vec_map["france"]
italy = word_to_vec_map["italy"]
paris = word_to_vec_map["paris"]
rome = word_to_vec_map["rome"]
 
print("cosine_similarity(father, mother) = ", cosine_similarity(father, mother)) #0.8656661174315731
print("cosine_similarity(ball, crocodile) = ",cosine_similarity(ball, crocodile)) #0.15206575219836116
print("cosine_similarity(france - paris, rome - italy) = ",cosine_similarity(france - paris, rome - italy)) #-0.7056238800453

def complete_analogy(word_a, word_b, word_c, word_to_vec_map):
    """
    Performs the word analogy task as explained : a is to b as c is to ____. 

    man is to woman as king is to queen using eb−ea ≈ ed−ec

    Arguments:
    word_a -- a word, string
    word_b -- a word, string
    word_c -- a word, string
    word_to_vec_map -- dictionary that maps words to their corresponding vectors. 
    
    Returns:
    best_word --  the word such that v_b - v_a is close to v_best_word - v_c, as measured by cosine similarity
    """
    
    # convert words to lower case
    word_a, word_b, word_c = word_a.lower(), word_b.lower(), word_c.lower()
    
    ### START CODE HERE ###
    # Get the word embeddings v_a, v_b and v_c (≈1-3 lines)
    e_a, e_b, e_c = word_to_vec_map[word_a], word_to_vec_map[word_b], word_to_vec_map[word_c]
    ### END CODE HERE ###
    
    words = word_to_vec_map.keys()
    max_cosine_sim = -100              # Initialize max_cosine_sim to a large negative number
    best_word = None                   # Initialize best_word with None, it will help keep track of the word to output

    # loop over the whole word vector set
    for w in words:        
        # to avoid best_word being one of the input words, pass on them.
        if w in [word_a, word_b, word_c] :
            continue
        
        ### START CODE HERE ###
        # Compute cosine similarity between the vector (e_b - e_a) and the vector ((w's vector representation) - e_c)  (≈1 line)
        cosine_sim = cosine_similarity(e_b - e_a, word_to_vec_map[w] - e_c)
        
        # If the cosine_sim is more than the max_cosine_sim seen so far,
            # then: set the new max_cosine_sim to the current cosine_sim and the best_word to the current word (≈3 lines)
        if cosine_sim > max_cosine_sim:
            max_cosine_sim = cosine_sim
            best_word = w
        ### END CODE HERE ###
        
    return best_word

triads_to_try = [('italy', 'italian', 'spain'), ('india', 'delhi', 'japan'), ('man', 'woman', 'boy'), ('small', 'smaller', 'large')]
for triad in triads_to_try:
    print ('{} -> {} :: {} -> {}'.format( *triad, complete_analogy(*triad, word_to_vec_map)))
'''
italy -> italian :: spain -> spanish
india -> delhi :: japan -> osaka
man -> woman :: boy -> girl
small -> smaller :: large -> larger
'''
############### DE-BIASING #####################
g = word_to_vec_map['woman'] - word_to_vec_map['man']
print(g) # vector $g$ roughly encodes the concept of "gender".

print ('List of names and their similarities with constructed vector:')
# girls and boys name
name_list = ['john', 'marie', 'sophie', 'ronaldo', 'priya', 'rahul', 'danielle', 'reza', 'katy', 'yasmin']
for w in name_list:
    print (w, cosine_similarity(word_to_vec_map[w], g))

'''
john -0.22835017436721872
marie 0.24537338638963357
sophie 0.20268358510223194
ronaldo -0.3328964498414265
priya 0.13922857114427084
rahul -0.0639072688424743
danielle 0.14913149265020167
reza -0.08192654617678002
katy 0.18688587375866103
yasmin 0.21136825959729247
'''
#female first names tend to have a positive cosine similarity with our constructed vector $g$, while male first names tend to have a negative cosine similarity. 

print('Other words and their similarities:')
word_list = ['lipstick', 'guns', 'science', 'arts', 'literature', 'warrior','doctor', 'tree', 'receptionist', 
             'technology',  'fashion', 'teacher', 'engineer', 'pilot', 'computer', 'singer']
for w in word_list:
    print (w, cosine_similarity(word_to_vec_map[w], g))
#results reflect certain unhealthy gender stereotypes. For example, "computer" is closer to "man" while "literature" is closer to "woman"
'''
lipstick 0.18037245461893886
guns -0.09964446323350887
science -0.02147576571900459
arts 0.01484674744156461
literature 0.08261854474431136
warrior -0.15634200481756028
doctor 0.10942282324077059
tree -0.08868359642037957
receptionist 0.2806875926160281
technology -0.14474526940138327
fashion 0.08097436821066459
teacher 0.1523369596791014
engineer -0.12300012058033458
pilot -0.04113394172314754
computer -0.11545715478097537
singer 0.11372642801434334
'''