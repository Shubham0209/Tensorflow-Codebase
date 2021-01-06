import numpy as np
from w2v_utils import *
words, word_to_vec_map = read_glove_vecs('glove.6B.100d.txt')
from collections import Counter
import nltk
from nltk.corpus import stopwords

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

# def extractEntityNames(tree, _entityNames=None):
#     """
#     Creates a local list to hold nodes of tree passed through, extracting named
#     entities from the chunked sentences.

#     Used for reference: https://gist.github.com/onyxfish/322906
#     """
#     if _entityNames is None:
#         _entityNames = []
#     try:
#         label = tree.label()
#     except AttributeError:
#         pass
#     else:
#         if label == 'NE':
#             _entityNames.append(' '.join([child[0] for child in tree]))
#         else:
#             for child in tree:
#                 extractEntityNames(child, _entityNames=_entityNames)
#     return _entityNames


# def buildDict(chunkedSentences, _entityNames=None):
#     """
#     Uses the global entity list, creating a new dictionary with the properties
#     extended by the local list, without overwriting.

#     Used for reference: https://gist.github.com/onyxfish/322906
#     """
#     if _entityNames is None:
#         _entityNames = []

#     for tree in chunkedSentences:
#         extractEntityNames(tree, _entityNames=_entityNames)

#     return _entityNames
# def chunkSentences(text):
#     """
#     Parses text into parts of speech tagged with parts of speech labels.
#     """
#     sentences = nltk.sent_tokenize(text)
#     tokenizedSentences = [nltk.word_tokenize(sentence)
#                           for sentence in sentences]
#     taggedSentences = [nltk.pos_tag(sentence)
#                        for sentence in tokenizedSentences]
#     chunkedSentences = nltk.ne_chunk_sents(taggedSentences, binary=True)
#     return chunkedSentences

# def removeStopwords(entityNames, customStopWords=None):
#     """
#     Brings in stopwords and custom stopwords to filter mismatches out.
#     """
#     # Memoize custom stop words
#     for name in entityNames:
#         if name in stopwords.words('english'):
#             entityNames.remove(name)
# def getMajorCharacters(entityNames):
#     """
#     Adds names to the major character list if they appear frequently.
#     """
#     return {name for name in entityNames if entityNames.count(name) > 5}
# def readText():
#     """
#     Reads the text from a text file.
#     """
#     with open("Harry Potter and the Sorcerer.txt", "r") as f:
#         text = f.read()
#     return text
# text = readText()
# chunkedSentences = chunkSentences(text)
# entityNames = (buildDict(chunkedSentences))
# # print(entityNames)
# removeStopwords(entityNames)
# majorCharacters = list(getMajorCharacters(entityNames))
# print(majorCharacters)


g = word_to_vec_map['woman'] - word_to_vec_map['man']# vector $g$ roughly encodes the concept of "gender".

majorCharacters = ['CHAPTER', 'Fluffy', 'Madam Hooch', 'Malfoy', 'Mr. Potter', 'Professor Flitwick', 'Hermione Granger', 'Snape', 'Lee Jordan', 'Wood', 'Madam Malkin', 'Percy', 'Bloody Baron', 'Come', 'Gryffindor', 'Diagon Alley', 'Mom', 'Crabbe', 'Firenze', 'Harry Potter', 'Privet Drive', 'Peeves', 'George', 'Voldemort', 'Stone', 'Albus Dumbledore', 'Uncle Vernon', 'Weasley', 'High Table', 'Look', 'Muggle', 'Piers', 'Hermione', 'Bane', 'Seamus', 'Charlie', 'Filch', 'Harry', 'Slytherin', 'Griphook', 'Ron', 'Norbert', 'Professor', 'Seamus Finnigan', 'Quidditch', 'Leaky Cauldron', 'Hufflepuff', 'Mr. Dursley', 'McGonagall', 'Snitch', 'Everyone', 'Fat Lady', 'Gringotts', 'Fang', 'Aunt Petunia', 'London', 'Dursleys', 'Flint', 'Mr. Ollivander', 'Sorcerer', 'Magic', 'Hagrid', 'Muggles', 'George Weasley', 'Draco Malfoy', 'Professor McGonagall', 'Hedwig', 'Weasleys', 'Dumbledore', 'Dark Arts', 'Potter', 'Madam Pomfrey', 'Seeker', 'Hogwarts', 'Nicolas Flamel', 'Dudley', 'Goyle', 'Quaffle', 'Slytherins', 'Quirrell', 'Neville', 'Ronan', 'Ravenclaw', 'Flamel', 'Great Hall', 'Fred', 'Dean']
for w in majorCharacters:
    print(w.split()[0])
print ('List of names and their gender:')
for w in majorCharacters:
    if w.split()[0].lower() in words:
        if (cosine_similarity(word_to_vec_map[w.split()[0].lower()], g) < 0):
            print (w, 'Male')
        else:
            print (w, 'Female')
    else:
        print(w, 'Nan')