import numpy
import os
import logging

from gensim import corpora
from gensim import models
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler

from preprocess.preprocess import get_data
from preprocess.stopwords import Utils

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# vietnamese stop words list
stopwords = Utils().load_stopwords()


# Tokenize and remove stopwords
def remove_stopwords(texts):
    texts_processed = []
    for doc in texts:
        spacy_doc = []
        for word in doc.split(' '):
            if word not in stopwords:
                spacy_doc.append(word)
        texts_processed.append(spacy_doc)
    return texts_processed


# Create dictionary
def create_dictionary(data):
    return corpora.Dictionary(data)


# Create corpus
def create_corpus(id2word, data):
    return [id2word.doc2bow(text) for text in data]


# implement
# prepare data
data_path = os.path.dirname(os.path.realpath(os.getcwd()))
data_path = os.path.join(data_path, 'data')

X_train, y_train = get_data(os.path.join(data_path, 'data_test', 'test_test'))

# texts
texts = remove_stopwords(X_train)

# Get corpus
train_id2word = create_dictionary(texts)
train_corpus = create_corpus(train_id2word, texts)
bigram_train = texts

# model
lda_train = models.ldamulticore.LdaMulticore(corpus=train_corpus, id2word=train_id2word,
                                             num_topics=10, passes=5, chunksize=100,
                                             per_word_topics=True, workers=3,
                                             eval_every=1)
# lda_train.save('lda_train.model')


topic = lda_train.print_topics(num_topics=10, num_words=10)

train_vecs = []
for i in range(len(X_train)):
    top_topics = lda_train.get_document_topics(train_corpus[i], minimum_probability=0.0)
    topic_vec = [top_topics[i][1] for i in range(10)]
    train_vecs.append(topic_vec)
    print(topic_vec)
    print()

# predict
X = numpy.array(train_vecs)
y = numpy.array(y_train)

X_train = StandardScaler().fit_transform(X)

sgd_huber = linear_model.SGDClassifier(
    max_iter=1000,
    tol=1e-3,
    alpha=10,
    loss='modified_huber',
    class_weight='balanced'
).fit(X_train, y)

y_pred = sgd_huber.predict(X_train[0:12])
# cv_svcsgd_f1.append(f1_score(y_pred, average='binary'))

print(y_pred)
print(y[0:12])
