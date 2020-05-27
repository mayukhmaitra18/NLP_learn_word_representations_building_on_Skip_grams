import os
import pickle
import numpy as np

model_path = './models/'
loss_model = 'cross_entropy'
#loss_model = 'nce'

model_filepath = os.path.join(model_path, 'word2vec_%s.model'%(loss_model))

dictionary, steps, embeddings = pickle.load(open(model_filepath, 'rb'))

"""
==========================================================================

Write code to evaluate a relation between pairs of words.
You can access your trained model via dictionary and embeddings.
dictionary[word] will give you word_id
and embeddings[word_id] will return the embedding for that word.

word_id = dictionary[word]
v1 = embeddings[word_id]

or simply

v1 = embeddings[dictionary[word_id]]

==========================================================================
"""


def cosine_similarity_distance(point1, point2):
    dotprod = np.dot(point1, point2)
    sqrtprod = np.sqrt(np.dot(point1, point1)) * np.sqrt(np.dot(point2, point2))
    return 1 - float(dotprod / sqrtprod)

def max_val(list_name):
    max = -99999999999
    for i in list_name:
        if i > max:
            max = i
    return max

def min_val(list_name):
    min = 99999999999
    for i in list_name:
        if i < min:
            min = i
    return min

final_ans = ""
output_file = open('word_analogy_ce.txt', 'w')
with open('word_analogy_dev.txt') as f:
    for line in f:
        line.strip()
        set_of_pairs = line.split("||")[1]
        words = set_of_pairs.strip().split(",")
        similarity = []
        i = 0
        while i < len(words):
            wrd_1, wrd_2 = words[i].strip().split(":")
            wrd_1 = wrd_1.strip('"')
            wrd_2 = wrd_2.strip('"')

            wrd_1, wrd_2 = wrd_1[0],wrd_2[1]
            embed_1,embed_2 = embeddings[dictionary[wrd_1]], embeddings[dictionary[wrd_2]]
            similarity.append(cosine_similarity_distance(embed_1,embed_2))
            i += 1
        final_ans += set_of_pairs.strip().replace(",", " ") + " " + words[similarity.index(max_val(similarity))].strip() + " " + words[similarity.index(min_val(similarity))] + "\n"

output_file.write(final_ans)
output_file.close()