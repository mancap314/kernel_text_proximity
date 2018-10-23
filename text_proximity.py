import os, re, time, itertools
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
    print(f'matrices for articles computed in {duration}s')

    for article in articles:
        article_words = re.findall(r'\w+', article.lower())
        article_matrix = None
        for i, word in enumerate(article_words):
            if word in words_vector:
                vector = words_vector[word]
                vector = vector.reshape((vector.shape[0], 1))
                if article_matrix is None:
                    article_matrix = vector
                else:
                    article_matrix = np.concatenate((article_matrix, vector), 1)
        result.append(article_matrix)
    return result


def append_word_distance(article_matrix_list, eps):
    for i, article_matrix in enumerate(article_matrix_list):
        n_words = article_matrix.shape[1]
        vec = np.array([eps * i / n_words for i in range(n_words)])
        vec = vec.reshape((1, n_words))
        article_matrix_list[i] = np.concatenate((article_matrix, vec), 0)
    return article_matrix_list


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
    a = sq_dists.shape.index(min(sq_dists.shape))
    return np.min(sq_dists, axis=a)


def get_score(sq_distances, l=.1):
    return np.sum(np.exp(-sq_distances / l ** 2))


def get_df_article_matrix_list(df, col_0, col_1):
    articles_list = df[col_0].tolist() + df[col_1].tolist()
    return get_articles_matrices(articles_list)


def get_df_distances_list(article_matrix_list, eps):
    n_pairs = int(len(article_matrix_list) / 2)
    article_matrix_list = append_word_distance(article_matrix_list, eps)
    sq_distances_list = []
    for i in range(n_pairs):
        sq_distances_list.append(get_sq_distances(article_matrix_list[i], article_matrix_list[n_pairs + i]))
    return sq_distances_list


def get_df_score(sq_distances_list, l):
    scores = [get_score(sq_distances, l) for sq_distances in sq_distances_list]
    min_score, max_score = min(scores), max(scores)
    scores = [(s - min_score) / (max_score - min_score) for s in scores]
    return pd.Series(scores)


def calibrate_l(df, col_0, col_1, col_score, l_values, eps_values):
    article_matrix_list = get_df_article_matrix_list(df, col_0, col_1)
    ap_score_best, l_best, eps_best = -1, None, None
    for l, eps in itertools.product(*[l_values, eps_values]):
        sq_distances_list = get_df_distances_list(article_matrix_list, eps)
        scores = get_df_score(sq_distances_list, l)
        ap_score = average_precision_score(df[col_score], scores)
        if ap_score > ap_score_best:
            ap_score_best = ap_score
            l_best = l
            eps_best = eps

    return l_best, eps_best, ap_score_best


def evaluate(df, col_0, col_1, l, eps):
    article_matrix_list = get_df_article_matrix_list(df, col_0, col_1)
    sq_distances_list = get_df_distances_list(article_matrix_list, eps)
    return get_df_score(sq_distances_list, l)



# # Example calibration
# df = pd.read_csv(os.path.join('data', 'questions.csv')).iloc[:1000]
# l_best, eps_best, ap_score_best = calibrate_l(df, 'question1', 'question2', 'is_duplicate', np.linspace(.0001, .2, 100), np.linspace(.001, 2, 100))
# print('l_best: {}, eps_best: {}, corresponding best average precision: {}'.format(l_best, eps_best, ap_score_best))

# # Example evaluation
# df = pd.read_csv(os.path.join('data', 'questions.csv')).iloc[:10]
# scores = evaluate(df, 'question1', 'question2', l=.15)
# print(scores)
# print(df['is_duplicate'])
