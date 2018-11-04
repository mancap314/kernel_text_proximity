import os, re, pickle

NLP_RESOURCE_DIRECTORY = os.path.join(os.path.expanduser('~'), 'nlp-resources')
REUTERS_SUBDIRECTORY = 'reuters'
LOCAL_DATA_DIRECTORY = 'data'

if not os.path.isdir(LOCAL_DATA_DIRECTORY):
    os.makedirs(LOCAL_DATA_DIRECTORY)


def get_articles():
    """
    extract articles from (unpacked) https://archive.ics.uci.edu/ml/machine-learning-databases/reuters21578-mld/reuters21578.tar.gz
    :return: list of articles
    """
    article_list_filepath = os.path.join(LOCAL_DATA_DIRECTORY, 'article_list.pkl')
    if os.path.isfile(article_list_filepath):
        with open(article_list_filepath, 'rb') as handle:
            article_list = pickle.load(handle)
            return article_list

    article_list = []
    for filename in os.listdir(os.path.join(NLP_RESOURCE_DIRECTORY, REUTERS_SUBDIRECTORY)):
        if filename.endswith('.sgm'):
            filepath = os.path.join(*[NLP_RESOURCE_DIRECTORY, REUTERS_SUBDIRECTORY, filename])
            with open(filepath, 'rb') as f:
                content = f.read()
            content = str(content.decode('utf-8','ignore').encode('utf-8'))
            for article in re.findall("<BODY>(.*?)</BODY>", content):
                article = article.replace('\\n', ' ')
                article = re.sub(r'&.*;', ' ', article)
                article = re.sub(r' +', ' ', article)
                article = re.sub(r' Reuter $', '', article)
                article_list.append(article)

    with open(article_list_filepath, 'wb') as handle:
        pickle.dump(article_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return article_list

# # Example
# articles = get_articles()
# print(f'n_articles: {len(articles)}\narticle #1000:\n{articles[1000]}\n')