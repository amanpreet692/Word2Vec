import os
import pickle
import sys
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

model_path = './models/'
loss_model = 'cross_entropy'
#loss_model = 'nce'

EXAMPLE_SEPARATOR = "||"
PAIR_SEPARATOR = ","
KEY_VALUE_SEPARATOR = ":"

model_filepath = os.path.join(model_path, 'word2vec_%s.model' % (loss_model))
file_name = sys.argv[1]
dictionary, steps, embeddings = pickle.load(open(model_filepath, 'rb'))
data_file = '%s.txt' % (file_name)
prediction_file = '%s_predictions_%s.txt' % (file_name, loss_model)
result_file = open(prediction_file, "w")

with open(data_file, "r") as lines:
    for line in lines:
        line = line.replace("\"", "")
        line = line.replace("\n", "")
        line_parts = line.split(EXAMPLE_SEPARATOR)
        examples = line_parts[0].split(PAIR_SEPARATOR)
        choices = line_parts[1].split(PAIR_SEPARATOR)

        example_distance_vectors = []
        for example in examples:
            key_value = example.split(KEY_VALUE_SEPARATOR)
            key_vector = embeddings[dictionary[example[0]]]
            value_vector = embeddings[dictionary[example[1]]]
            dist_vector = value_vector - key_vector
            example_distance_vectors.append(dist_vector)

        choice_dist_vectors = []
        for choice in choices:
            key_value = choice.split(KEY_VALUE_SEPARATOR)
            key_vector = embeddings[dictionary[choice[0]]]
            value_vector = embeddings[dictionary[choice[1]]]
            dist_vector = value_vector - key_vector
            choice_dist_vectors.append(dist_vector)
            result_file.write('"%s" ' % choice)
        cosine_sims = []
        for dist in choice_dist_vectors:
            cosine_sims.append(np.average(cosine_similarity(np.reshape(dist, [1, -1]), example_distance_vectors)))
        result_file.write('"%s" "%s"\n' % (choices[np.argmin(cosine_sims)], choices[np.argmax(cosine_sims)]))
    print("predictions written to %s" % prediction_file)

result_file.close()

# Words Similarity in generated model
print("Words similarity task")

word_sim_dict = {}

for word in ['first', 'american', 'would']:
    word_embedding = embeddings[dictionary[word]]
    top_k = 20
    word_sims = np.asarray(cosine_similarity(np.reshape(word_embedding, [1, -1]), embeddings)).reshape(-1)
    top_k_word_sims = (-word_sims).argsort()[:top_k + 1]
    sim_words_list = []
    for top_word in top_k_word_sims:
        for key_word in dictionary.keys():
            if dictionary[key_word] == top_word:
                if key_word == word:
                    continue
                sim_words_list.append(key_word)
    word_sim_dict[word] = sim_words_list
for key in word_sim_dict.keys():
    print("{}:{}".format(key, word_sim_dict[key]))

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


def similarity(x, y):
    return cosine_similarity(x, y)
