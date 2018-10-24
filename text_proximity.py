import os, re, time, itertools, operator, nltk
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def get_stop_words(language='english'):
    try:
        stop_words = nltk.corpus.stopwords.words(language)
    except:
        nltk.download('stopwords')
        stop_words = nltk.corpus.stopwords.words(language)
    return stop_words


def get_words_vector(words, word_embedding_file_path):
    """
    builds a dictionary word -> vector for the words where a vector exisits
    :param words: list, words to look-up for
    :return: a 'word' -> vec (np.array) dictionary
    """
    words = [str(word).lower() for word in words]
    res = {}
    with open(word_embedding_file_path) as word_embedding_file:
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


def get_articles_matrices(articles, word_embedding_file_path, language='english'):
    result = []
    concatenation = ' '.join(*[articles]).lower()
    word_list = re.findall(r'\w+', concatenation)
    word_list = list(set(word_list))
    stop_words = get_stop_words(language)
    word_list = [word for word in word_list if word not in stop_words]
    t0 = time.time()
    # Get their vectorization
    words_vector = get_words_vector(word_list, word_embedding_file_path)
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
        if article_matrix is None:
            continue
        n_words = article_matrix.shape[1]
        vec = np.array([eps * i / n_words for i in range(n_words)])
        vec = vec.reshape((1, n_words))
        article_matrix_list[i] = np.concatenate((article_matrix, vec), 0)
    return article_matrix_list


def get_sq_distances(matrix_0, matrix_1):
    if matrix_0 is None or matrix_1 is None:
        return None
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
    if sq_distances is None:
        return -1
    return np.mean(np.exp(-sq_distances / l ** 2))  # mean instead of sum


def get_df_article_matrix_list(df, col_0, col_1, word_embedding_file_path, language='english'):
    articles_list = df[col_0].tolist() + df[col_1].tolist()
    return get_articles_matrices(articles_list, word_embedding_file_path, language)


def get_df_distances_list(article_matrix_list, eps):
    n_pairs = int(len(article_matrix_list) / 2)
    article_matrix_list = append_word_distance(article_matrix_list, eps)
    sq_distances_list = []
    for i in range(n_pairs):
        sq_distances_list.append(get_sq_distances(article_matrix_list[i], article_matrix_list[n_pairs + i]))
    return sq_distances_list


def get_df_score(sq_distances_list, l):
    scores = [get_score(sq_distances, l) for sq_distances in sq_distances_list]
    min_score, max_score = min([s for s in scores if s > 0]), max([s for s in scores if s > 0])
    scores = [(s - min_score) / (max_score - min_score) for s in scores]
    return pd.Series(scores)


def calibrate_l(df, col_0, col_1, col_score,  word_embedding_file_path, l_values, eps_values, prop=1, language='english'):
    article_matrix_list = get_df_article_matrix_list(df, col_0, col_1,  word_embedding_file_path, language)
    ap_score_best, l_best, eps_best = -1, None, None
    values_to_test = list(itertools.product(*[l_values, eps_values]))
    indices_to_test = np.random.choice(len(values_to_test), size=int(round(prop * len(values_to_test))), replace=False)
    indices_to_test = [int(ind) for ind in indices_to_test]
    values_to_test = list(operator.itemgetter(*indices_to_test)(values_to_test))

    for l, eps in values_to_test:
        sq_distances_list = get_df_distances_list(article_matrix_list, eps)
        scores = get_df_score(sq_distances_list, l)
        ap_score = average_precision_score(df[col_score][scores >= 0], scores[scores >= 0])
        if ap_score > ap_score_best:
            ap_score_best = ap_score
            l_best = l
            eps_best = eps
            print('better parameters found: l_best={}, eps_best={}, corresponding best average precision: {}'.format(l_best, eps_best, ap_score_best))

    return l_best, eps_best, ap_score_best


def evaluate(df, col_0, col_1, l, eps, word_embedding_file_path, language='english'):
    article_matrix_list = get_df_article_matrix_list(df, col_0, col_1, word_embedding_file_path, language)
    sq_distances_list = get_df_distances_list(article_matrix_list, eps)
    return get_df_score(sq_distances_list, l)


# TODO: test handling of empty (None) article matrix

# # Define embedding file
# NLP_RESOURCE_DIRECTORY = os.path.join(os.path.expanduser('~'), 'nlp-resources')
# word_embedding_file_path = os.path.join(NLP_RESOURCE_DIRECTORY, 'glove.840B.300d.txt')
# # Example calibration
# df = pd.read_csv(os.path.join('data', 'questions.csv')).iloc[:1000]
# l_best, eps_best, ap_score_best = calibrate_l(df, 'question1', 'question2', 'is_duplicate', word_embedding_file_path, np.linspace(.0001, .2, 100), np.linspace(.5, 1.5, 20), .1)
# # l_best, eps_best, ap_score_best = calibrate_l(df, 'question1', 'question2', 'is_duplicate', np.linspace(.0001, .2, 100), [0], 1)
# print('l_best: {}, eps_best: {}, corresponding best average precision: {}'.format(l_best, eps_best, ap_score_best))

# # Example evaluation
# df = pd.read_csv(os.path.join('data', 'questions.csv')).iloc[:10]
# scores = evaluate(df, 'question1', 'question2', l=.15)
# print(scores)
# print(df['is_duplicate'])
