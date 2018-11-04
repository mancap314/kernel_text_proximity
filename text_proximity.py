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


def get_relevant_embeddings(articles, word_embedding_file_path, language='english', normalize=True):
    """
    builds a dictionary word -> vector for the words where a vector exisits
    :param words: list, words to look-up for
    :return: a 'word' -> vec (np.array) dictionary
    """
    print('computing relevant word embeddings...')
    t0 = time.time()
    concatenation = ' '.join(*[articles]).lower()
    words = re.findall(r'\w+', concatenation)
    words = list(set(words))
    stop_words = get_stop_words(language)
    words = [word for word in words if word not in stop_words]
    words = [str(word).lower() for word in words]
    relevant_embeddings = {}
    with open(word_embedding_file_path) as word_embedding_file:
        # Go through the file lines until word found
        for line in word_embedding_file:
            current_word = line.split(' ')[0].lower()
            if current_word in words:
                vec = line.split(' ')[1:]
                vec = np.array([float(value) for value in vec])
                if normalize:
                    vec = vec / np.sum(vec)
                relevant_embeddings[current_word] = vec
                words.remove(current_word)
            if len(words) == 0:
                break
    print('relevant word embeddings computed in {}s'.format(round(time.time() - t0, 1)))
    return relevant_embeddings


def get_articles_matrices(articles, relevant_embeddings):
    result = []
    for article in articles:
        article_words = re.findall(r'\w+', article.lower())
        article_matrix = None
        for i, word in enumerate(article_words):
            if word in relevant_embeddings:
                vector =relevant_embeddings[word]
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


def get_score(sq_distances, l=.1, kernel='gaussian'):
    if sq_distances is None:
        return -1
    if kernel == 'gaussian':
        return np.mean(np.exp(-sq_distances / l ** 2))  # mean instead of sum
    if kernel == 'inverse':
        return np.mean(np.minimum(l / sq_distances, 1))
    if kernel == 'triangle':
        return np.mean(np.maximum(1 - l * np.abs(sq_distances), 0))
    if kernel == 'epanechnikov':
        return np.mean(np.maximum(1 - l ** 2 * np.square(sq_distances), 0))
    if kernel == 'quadratic':
        return np.mean(np.maximum(np.square(1 - l ** 2 * np.square(sq_distances)), 0))
    if kernel == 'cubic':
        return np.mean(np.maximum(np.power(1 - l ** 2 * np.square(sq_distances), 3), 0))
    if kernel == 'circular':
        return np.mean(np.maximum(np.cos(np.pi / 2 * l * sq_distances), 0))



def get_scores_pairs(articles_0, articles_1, eps, l, relevant_embeddings, kernel):
    scores = []
    for i in range(len(articles_0)):
        matrices = get_articles_matrices([articles_0[i], articles_1[i]], relevant_embeddings)
        append_word_distance(matrices, eps)
        matrix_0, matrix_1 = matrices[0], matrices[1]
        sq_distances = get_sq_distances(matrix_0, matrix_1)
        score = get_score(sq_distances, l, kernel)
        scores.append(score)
    min_score, max_score = min(scores), max(scores)
    scores = pd.Series([(s - min_score) / (max_score - min_score) for s in scores])
    return scores


def calibrate(articles_0, articles_1, true_values, word_embedding_file_path, l_values, eps_values, kernels=['gaussian'], prop=1, language='english', normalize=True):
    assert len(articles_0) == len(articles_1), 'article_0 (length={}) must have same length than article_1 (length={})'.format(len(articles_0), len(articles_1))
    relevant_embeddings = get_relevant_embeddings(articles_0 + articles_1, word_embedding_file_path, language, normalize)

    print('searching for best l and eps values...')
    t0 = time.time()
    y_true = pd.Series(true_values)
    ap_score_best, kernel_best, l_best, eps_best = -1, None, None, None
    values_to_test = list(itertools.product(*[kernels, l_values, eps_values]))
    indices_to_test = np.random.choice(len(values_to_test), size=int(round(prop * len(values_to_test))), replace=False)
    indices_to_test = [int(ind) for ind in indices_to_test]
    values_to_test = list(operator.itemgetter(*indices_to_test)(values_to_test))

    for kernel, l, eps in values_to_test:
        scores = get_scores_pairs(articles_0, articles_1, eps, l, relevant_embeddings, kernel)
        ap_score = average_precision_score(y_true[scores >= 0], scores[scores >= 0])
        if ap_score > ap_score_best:
            ap_score_best = ap_score
            kernel_best = kernel
            l_best = l
            eps_best = eps
            print('better parameters found: kernel=\'{}\', l_best={}, eps_best={}, corresponding best average precision: {}'.format(kernel_best, l_best, eps_best, ap_score_best))

    print('best l and eps values out of {} combinations from {} pairs computed in {}s'.format(len(values_to_test), len(articles_0), round(time.time() - t0, 1)))
    return kernel_best, l_best, eps_best, ap_score_best


def evaluate(articles_0, articles_1, l, eps, word_embedding_file_path, kernel='gaussian', language='english', normalize=False):
    assert len(articles_0) == len(articles_1), 'article_0 (length={}) must have same length than article_1 (length={})'.format(len(articles_0), len(articles_1))
    relevant_embeddings = get_relevant_embeddings(articles_0 + articles_1, word_embedding_file_path, language, normalize)
    print('computing pair scores...')
    t0 = time.time()
    scores = get_scores_pairs(articles_0, articles_1, eps, l, relevant_embeddings, kernel)
    print('scores for {} pairs computed in {}s'.format(len(articles_0), round(time.time() - t0, 1)))
    return scores


# TODO: test handling of empty (None) article matrix

# # Define embedding file
# NLP_RESOURCE_DIRECTORY = os.path.join(os.path.expanduser('~'), 'nlp-resources')
# word_embedding_file_path = os.path.join(NLP_RESOURCE_DIRECTORY, 'glove.840B.300d.txt')
#
# # Example calibration
# df = pd.read_csv(os.path.join('data', 'questions.csv')).iloc[:1000].dropna().reset_index(drop=True)
# kernel_best, l_best, eps_best, ap_score_best = calibrate(df['question1'].tolist(),
#                                                          df['question2'].tolist(),
#                                                          df['is_duplicate'],
#                                                          word_embedding_file_path,
#                                                          np.linspace(.00001, .005, 100),
#                                                          np.linspace(.1, 1.5, 45),
#                                                          ['gaussian', 'inverse', 'triangle', 'quadratic', 'cubic', 'circular'],
#                                                          .3)
# l_best, eps_best, ap_score_best = calibrate_l(df, 'question1', 'question2', 'is_duplicate', np.linspace(.0001, .2, 100), [0], 1)
# print('kernel_best: {}, l_best: {}, eps_best: {}, corresponding best average precision: {}'.format(kernel_best, l_best, eps_best, ap_score_best))
# # better parameters found: kernel='inverse', l_best=0.0031350505050505057, eps_best=0.19545454545454544, corresponding best average precision: 0.5657050176417399

# # Example evaluation
# df = pd.read_csv(os.path.join('data', 'questions.csv')).dropna().reset_index(drop=True)
# scores = evaluate(df['question1'].tolist(), df['question2'].tolist(), l=0.0001, eps=.55, word_embedding_file_path=word_embedding_file_path)
# ap_score = average_precision_score(df['is_duplicate'][scores >= 0], scores[scores >= 0])
# print('ap score: {}'.format(ap_score))
