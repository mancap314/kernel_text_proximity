import os, re, time
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

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
                # TODO: find better default distance
                vector = np.append(words_vector[word], i / (len(article_words) - 1))
                vector = vector.reshape((vector.shape[0], 1))
                if article_matrix is None:
                    article_matrix = vector
                else:
                    article_matrix = np.concatenate((article_matrix, vector), 1)
        result.append(article_matrix)
    return result


def get_sq_distances(matrix_0, matrix_1):
    sq_dists = None
    for i in range(matrix_0.shape[1]):
        vector = matrix_0[:,i]
        vector = vector.reshape((vector.shape[0], 1))
        sq_dist = np.sum(np.square(matrix_1 - vector), 0)
        sq_dist = sq_dist.reshape((sq_dist.shape[0], 1))
        if sq_dists is None:
            sq_dists = sq_dist
        else:
            sq_dists = np.concatenate((sq_dists, sq_dist), 1)
    return sq_dists


def get_min_sq_distances(article_0, article_1):
    matrices = get_articles_matrices([article_0, article_1])
    sq_distances = get_sq_distances(matrices[0], matrices[1])
    a = sq_distances.shape.index(min(sq_distances.shape))
    return np.min(sq_distances, axis=a)


def get_score(sq_distances, l=.1):
    return np.sum(np.exp(-sq_distances / l ** 2))

# TODO: refactor calibrate and evaluate to externalize common actions
# TODO: get matrices for all articles at once (much faster)

def calibrate_l(df, col_0, col_1, col_score, l_values):
    sq_distances_list = []
    for i, row in df.iterrows():
        sq_distances_list.append(get_min_sq_distances(row[col_0], row[col_1]))

    ap_score_best, l_best = -1, None
    for l in l_values:
        scores = [get_score(sq_distances, l) for sq_distances in sq_distances_list]
        min_score, max_score = min(scores), max(scores)
        scores = [(s - min_score) / (max_score - min_score) for s in scores]
        scores = pd.Series(scores)
        ap_score = average_precision_score(df[col_score], scores)
        if ap_score > ap_score_best:
            ap_score_best = ap_score
            l_best = l

    return l_best, ap_score_best


def evaluate(df, col_0, col_1, l):
    sq_distances_list = []
    for i, row in df.iterrows():
        sq_distances_list.append(get_min_sq_distances(row[col_0], row[col_1]))

    scores = [get_score(sq_distances, l) for sq_distances in sq_distances_list]
    min_score, max_score = min(scores), max(scores)
    scores = [(s - min_score) / (max_score - min_score) for s in scores]
    scores = pd.Series(scores)
    return scores


# # Example score
# score = compare('tomorrow it will rain. Horrible weather is expected', 'in one day there will be bad weather. People wait for better times')
# print('score: {}'.format(score))

# # Example calibration
# df = pd.read_csv(os.path.join('data', 'questions.csv')).iloc[:10]
# l_best, ap_score_best = calibrate_l(df, 'question1', 'question2', 'is_duplicate', np.linspace(.001, 1, 100))
# print('l_best: {}, corresponding best average precision: {}'.format(l_best, ap_score_best))

# Example evaluation
df = pd.read_csv(os.path.join('data', 'questions.csv')).iloc[:10]
scores = evaluate(df, 'question1', 'question2', l=.15)
print(scores)
print(df['is_duplicate'])
