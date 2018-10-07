import os, re, time
from sklearn.neighbors.kde import KernelDensity
from sklearn.model_selection import GridSearchCV
import numpy as np
import utils

NLP_RESOURCE_DIRECTORY = os.path.join(os.path.expanduser('~'), 'nlp-resources')


def get_words_vector(words):
    """
    builds a dictionary word -> vector for the words where a vector exisits
    :param words: list, words to look-up for
    :return: a 'word' -> vec (np.array) dictionary
    """
    words = [str(word).lower() for word in words]
    res = {}
    with open(os.path.join(NLP_RESOURCE_DIRECTORY, 'glove.840B.300d.txt')) as word_embedding_file:
        # Go through the file lines until word found
        for line in word_embedding_file:
            current_word = line.split(' ')[0].lower()
            if current_word in words:
                vec = line.split(' ')[1:]
                vec = np.array([float(value) for value in vec])
                res[current_word] = vec
                words.remove(current_word)
            if len(words) == 0:
                break

    return res



def get_articles_matrices(articles):
    result = []
    concatenation = ' '.join(*[articles]).lower()
    word_list = re.findall(r'\w+', concatenation)
    word_list = list(set(word_list))
    t0 = time.time()
    # Get their vectorization
    words_vector = get_words_vector(word_list)
    duration = round(time.time() - t0, 1)
    print(f'word_vector computed in {duration}s')

    for article in articles:
        article_words = re.findall(r'\w+', article.lower())
        article_matrix = None
        for i, word in enumerate(article_words):
            if word in words_vector:
                vector = np.append(words_vector[word], i / (len(article_words) - 1))
                vector = vector.reshape((vector.shape[0], 1))
                if article_matrix is None:
                    article_matrix = vector
                else:
                    article_matrix = np.concatenate((article_matrix, vector), 1)
        result.append(article_matrix)
    return result


def compare(article_0, article_1):
    article_matrices = get_articles_matrices([article_0, article_1])
    params = {'bandwidth': np.logspace(-1, 1, 20)}

    grid = GridSearchCV(KernelDensity(), params, cv=5)
    grid.fit(article_matrices[0].T)

    kde_0 = grid.best_estimator_


    grid = GridSearchCV(KernelDensity(), params, cv=5)
    grid.fit(article_matrices[1].T)
    kde_1 = grid.best_estimator_
    new_data_1 = kde_1.sample(article_matrices[0].shape[1], random_state=0)

    score = kde_0.score(new_data_1)
    return score


# # similar article
# print(compare('tomorrow it will rain. Horrible weather is expected', 'in one day there will be bad weather. People wait for better times'))
# # -1087.2591809224195
#
# # unsimilar article
# print(compare('yesterday the president said he does not care at all. People are deceid', 'there was a big fire in the forest in greec, 1000 people were evacuated'))
# # -2758.4067761311753
#
# # the more negative, the more unsimilar, as expected