import re, time, itertools, operator, nltk, logging
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import average_precision_score
from hyperopt import fmin, tpe, hp


logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(levelname)-8s: %(asctime)s, %(name)-12s, %(funcName)s: %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


def set_logger_level(level):
    """level: a level of logging"""
    logger.setLevel(level)


def get_stop_words(language='english'):
    try:
        stop_words = nltk.corpus.stopwords.words(language)
    except:
        nltk.download('stopwords')
        stop_words = nltk.corpus.stopwords.words(language)
    return stop_words


def get_relevant_embeddings(articles, word_embedding_file_path, language='english', normalize=True):
    """
    get word embeddings for all the words in `articles` found in the file at `word_embedding_file_path` that are not
    stopwords in `language`
    :param articles: list of articles (texts, strings)
    :param word_embedding_file_path: path to the word embedding file (that has to be in GloVe format)
    :param language: string, any language for which nltk has a stopword list
    :param normalize: Boolean, if to normalize the word embedding vectors or not
    :return: dict word -> embedding_vector for all the words in `articles` having an embedding in word_embedding_file
    """
    logger.info('computing relevant word embeddings...')
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
    logger.info('relevant word embeddings computed in {}s'.format(round(time.time() - t0, 1)))
    return relevant_embeddings


def get_articles_matrices(articles, relevant_embeddings):
    """
    Computes 2d array for each article in `articles` where each colum is a word embedding from `elevant_enbeddings`
    :param articles: list of articles (texts, strings)
    :param relevant_embeddings: dict of word embedding vectors as returned by get_relevant_embeddings()
    :return: list of numpy ndarrays. For each ndarray, the columns are the sequence of word embedding vectors
    """
    result = []
    for article in articles:
        article_words = re.findall(r'\w+', article.lower())
        article_matrix = None
        for word in article_words:
            if word in relevant_embeddings:
                vector = relevant_embeddings[word]
                vector = vector.reshape((vector.shape[0], 1))
                if article_matrix is None:
                    article_matrix = vector
                else:
                    article_matrix = np.concatenate((article_matrix, vector), 1)
        result.append(article_matrix)
    return result


def append_word_distance(article_matrix_list, length):
    """
    Appends a coordinate to the word embeddings corresponding to the position of each word in the sequence of words
    :param article_matrix_list: list of numpy ndarrays as returned by get_articles_matrices()
    :param length: length of the 'box' where the words are embedded
    :return: list of numpy ndarray where the last coordinate of each column is the position of the word on the
    sequential axis
    """
    for i, article_matrix in enumerate(article_matrix_list):
        if article_matrix is None:
            continue
        n_words = article_matrix.shape[1]
        vec = np.array([length * i / n_words for i in range(n_words)])
        vec = vec.reshape((1, n_words))
        article_matrix_list[i] = np.concatenate((article_matrix, vec), 0)
    return article_matrix_list


def get_distances(matrix_0, matrix_1, method='euclidian'):
    """
    Computes distances between columns of `matrix_0` and `matrix_1`
    :param matrix_0: first numpy ndarray
    :param matrix_1: second numpy ndarray
    :param method: 'euclidian' or 'cosine'
    :return: numpy ndarray corresponding where x[i, j] = euclidian (or cosine) distance between column i of `matrix_0`
    and column j of `matrix_1` (or reversely if `matrix_1` has less columns than `matrix_0`)
    """
    if method == 'euclidian':
        distances = get_euclidian_distances(matrix_0, matrix_1)
    if method == 'cosine':
        distances = get_cosine_distances(matrix_0, matrix_1)
    return distances


def get_euclidian_distances(matrix_0, matrix_1):
    """
    Computes euclidian distances between columns of `matrix_0` and `matrix_1`
    :param matrix_0: first numpy ndarray
    :param matrix_1: second numpy ndarray
    :return: numpy ndarray corresponding where x[i, j] = euclidian distance between column i of `matrix_0`
    and column j of `matrix_1` (or reversely if `matrix_1` has less columns than `matrix_0`)
    """
    if matrix_0 is None or matrix_1 is None:
        return None
    if matrix_0.shape[1] > matrix_1.shape[1]:
        matrix_0, matrix_1 = matrix_1, matrix_0[0]
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
    return np.sqrt(sq_dists)


def get_cosine_distances(matrix_0, matrix_1):
    """
    Computes cosine distances between columns of `matrix_0` and `matrix_1`
    :param matrix_0: first numpy ndarray
    :param matrix_1: second numpy ndarray
    :return: numpy nd array where x[i, j] = cosine distance between column i of `matrix_0`
    and column j of `matrix_1` (or reversely if `matrix_1` has less columns than `matrix_0`)
    """
    if matrix_0 is None or matrix_1 is None:
        return None
    cos_dists = cosine_distances(matrix_0.T, matrix_1.T)
    if cos_dists.shape[0] < cos_dists.shape[1]:
        cos_dists = cos_dists.T
    return cos_dists



def get_score(distances, scale=.1, kernel='gaussian'):
    """
    Computes kernel value for each entry of `sq_distances`, sum them by colum (word) and takes the mean value
    :param sq_distances: numpy nd array as returned by get_sq_distances()
    :param scale: float, scale parameter of `kernel`
    :param kernel: one of ('gaussian', 'inverse', 'triangle', 'epanechnikov', 'quadratic', 'cubic', 'circular',
    'student'). Default: 'gaussian'
    :return: float corresponding to the mean of the kernel values by column
    """
    if distances is None:
        return -1
    eps = .0001
    if scale == 0:
        scale += eps
    if kernel in ['inverse', 'circular']:  # kernels with division by `sq_distances`
        distances[distances==0] = eps
    distances /= scale ** 2
    if kernel == 'gaussian':
        transfo = np.exp(-distances)
    if kernel == 'inverse':
        transfo = np.minimum(1 / np.square(distances), 1)
    if kernel == 'triangle':
        transfo = np.maximum(1 - np.square(distances), 0)
    if kernel == 'epanechnikov':
        transfo = np.maximum(1 - np.square(distances), 0)
    if kernel == 'quadratic':
        transfo = np.maximum(np.square(1 -np.square(distances)), 0)
    if kernel == 'cubic':
        transfo = np.maximum(np.power(1 - np.square(distances), 3), 0)
    if kernel == 'circular':
        transfo = np.maximum(np.cos(np.pi / (2 * distances)), 0)
    if kernel == 'student':
        transfo = 1 / (1 + distances ** 2)
    return np.mean(np.sum(transfo, 0))  # compute the sum by word (axis 0) and take the average


def get_scores_pairs(articles_0, articles_1, scale, length, relevant_embeddings, kernel, distance='euclidian', minmaxscale=True):
    """
    Computes  proximity between the articles in `articles_0` and the articles in `articles_1`
    :param articles_0: first list of articles (texts, strings)
    :param articles_1: second list of articles (texts, strings) with which the articles in `articles_0` will be compared
    :param scale: scale parameter of `kernel`
    :param length: length paramter of `kernel`
    :param relevant_embeddings: dict as returned by get_relevant_embeddings()
    :param kernel: string, one of the values accepted by get_score()
    :param minmaxscale: boolean, if to apply min_max scaling on the scores or not
    :return: pandas Series of proximity scores between the articles in `articles_0`and the àrticles in
    `articles_1
    """
    scores = []
    for i in range(len(articles_0)):
        matrices = get_articles_matrices([articles_0[i], articles_1[i]], relevant_embeddings)
        append_word_distance(matrices, length)
        matrix_0, matrix_1 = matrices[0], matrices[1]
        sq_distances = get_distances(matrix_0, matrix_1, distance)
        score = get_score(sq_distances, scale, kernel)
        scores.append(score)
    if minmaxscale:
        min_score, max_score = min(scores), max(scores)
        scores = [(s - min_score) / (max_score - min_score) for s in scores]
    return pd.Series(scores)


def calibrate(articles_0, articles_1, true_values, word_embedding_file_path, scale_values, length_values, normalize_values=[True, False], kernels=['gaussian'], distances=['euclidian'], prop=1, patience=1000, language='english'):
    """
    Gridsearch of hyperparameter values, possiblility to control the extent of the search through `prop` and `patience`
    :param articles_0: first list of articles (texts, strings)
    :param articles_1: second list of articles (texts, strings) with which the articles in `articles_0` will be compared
    :param true_values: pandas Series of labels (0 or 1) corresponding to text article_0 similar to article_1
    :param word_embedding_file_path: path to the file containing the word embeddings (in GloVe format)
    :param scale_values: numpy array containing the values to test for the scale parameter of the kernel
    :param length_values: numpy array containing the values to test for the length parameter of the kernel
    :param normalize_values: list (max. length=2) of booleans for the normalize parameter of the word embeddings
    :param kernels: list of names (strings) of kernels to test
    :param distances: type of distance to apply between word vectors, 'euclidian', 'cosine' or both
    :param prop: proportion of the hyperparameter combinations to test, sampled at random
    :param patience: number of tests without improvement before breaing the search
    :param language: string, language for which nltk has a list of stop words
    :return: kernel_best, scale_best, length_best, normalize_best, corresponding best average precision score
    """
    assert len(articles_0) == len(articles_1), 'article_0 (length={}) must have same length than article_1 (length={})'.format(len(articles_0), len(articles_1))

    logger.info('Searching for best scale and length values...')
    t0 = time.time()
    y_true = pd.Series(true_values)
    ap_score_best = -1
    values_to_test = list(itertools.product(*[kernels, scale_values, length_values, distances]))
    indices_to_test = np.random.choice(len(values_to_test), size=int(round(prop * len(values_to_test))), replace=False)
    indices_to_test = [int(ind) for ind in indices_to_test]
    values_to_test = list(operator.itemgetter(*indices_to_test)(values_to_test))

    for normalize in normalize_values:
        relevant_embeddings = get_relevant_embeddings(articles_0 + articles_1, word_embedding_file_path, language, normalize)
        n_unsuccessful_trials = 0
        for kernel, scale, length, distance in values_to_test:
            if n_unsuccessful_trials >= patience:
                logger.info('No better hyperparameters found after {} trials, break'.format(patience))
                break
            scores = get_scores_pairs(articles_0, articles_1, scale, length, relevant_embeddings, kernel, distance)
            ap_score = average_precision_score(y_true[scores >= 0], scores[scores >= 0])
            if ap_score > ap_score_best:
                n_unsuccessful_trials = 0
                ap_score_best = ap_score
                kernel_best = kernel
                scale_best = scale
                length_best = length
                normalize_best = normalize
                distance_best = distance
                logger.info('Better parameters found: kernel=\'{}\', scale_best={}, length_best={}, normalize={}, distance_best: {},  corresponding best average precision: {}'.format(kernel_best, scale_best, length_best, normalize, distance, ap_score_best))
            else:
                n_unsuccessful_trials += 1

    logger.info('Best scale and length values out of {} combinations from {} pairs computed in {}s'.format(len(values_to_test), len(articles_0), round(time.time() - t0, 1)))
    return kernel_best, scale_best, length_best, normalize_best, distance_best, ap_score_best


def calibrate_bayes(articles_0, articles_1, true_values, word_embedding_file_path,  scale_range=(0, 1), length_range=(0, .1), kernel='gaussian', distance='euclidian', max_evals=1000, normalize=True, language='english'):
    """
    Bayesian search of hyperparameter values for scale and length. `kernel` and `normalize` must be fixed.
    :param articles_0: first list of articles (texts, strings)
    :param articles_1: second list of articles (texts, strings) with which the articles in `articles_0` will be compared
    :param true_values: pandas Series of labels (0 or 1) corresponding to text article_0 similar to article_1
    :param word_embedding_file_path: path to the file containing the word embeddings (in GloVe format)
    :param scale_range: (min_scale, max_scale) of the kernel scale parameter. If sequence, will take the min, and max. values of it
    :param length_range: (min_length, max_length) of the kernel length parameter. If sequence, will take the min, and max. values of it
    :param kernel: name of the kernel to test, one of the kernel names accepted by get_score()
    :param distance: type of distance to apply between word vectors, 'euclidian', 'cosine' or both
    :param max_evals: maximal number of evaluations made by the Bayes optimizer
    :param normalize: boolean, notmalize word embedding vectors or not
    :param language: string, language for which nltk has a list of stop words
    :return: (best_scale, best_length)
    """
    logger.info('Starting Bayes hyperparameter optimization...')
    assert len(articles_0) == len(articles_1), 'article_0 (length={}) must have same length than article_1 (length={})'.format(len(articles_0), len(articles_1))
    t0 = time.time()

    relevant_embeddings = get_relevant_embeddings(articles_0 + articles_1, word_embedding_file_path, language, normalize)

    def objective(args):
        scale, length = args
        scores = get_scores_pairs(articles_0, articles_1, scale, length, relevant_embeddings, kernel, distance)
        ap_score = average_precision_score(true_values[scores >= 0], scores[scores >= 0])
        return 1 - ap_score

    space = [hp.uniform('scale', min(scale_range), max(scale_range)), hp.uniform('length', min(length_range), max(length_range))]
    best = fmin(objective, space=space, algo=tpe.suggest, max_evals=max_evals)
    logger.info('Bayes hyperparameter optimization finished in {}s'.format(round(time.time() - t0, 1)))
    return best


def evaluate(articles_0, articles_1, scale, length, word_embedding_file_path, kernel='gaussian', distance='euclidian', language='english', normalize=False):
    """
    Evaluates proximity between articles in `articles_0` and articles in `articles_1`, proximity scores min_max scaled
    :param articles_0: first list of articles (texts, strings)
    :param articles_1: second list of articles (texts, strings) with which the articles in `articles_0` will be compared
    :param scale: float, kernel scale parameter to use to proceed the evaluation
    :param length: float, kernel scale parameter to use to proceed the evaluation
    :param word_embedding_file_path: path to the file containing the word embeddings (in GloVe format)
    :param kernel: name of the kernel to test, one of the kernel names accepted by get_score()
    :param normalize: boolean, notmalize word embedding vectors or not
    :param language: string, language for which nltk has a list of stop words
    :return: pandas Series of min_max scaled proximity scores
    """
    assert len(articles_0) == len(articles_1), 'article_0 (length={}) must have same length than article_1 (length={})'.format(len(articles_0), len(articles_1))
    relevant_embeddings = get_relevant_embeddings(articles_0 + articles_1, word_embedding_file_path, language, normalize)
    logger.info('computing pair scores...')
    t0 = time.time()
    scores = get_scores_pairs(articles_0, articles_1, scale, length, relevant_embeddings, kernel, distance)
    logger.info('scores for {} pairs computed in {}s'.format(len(articles_0), round(time.time() - t0, 1)))
    return scores
