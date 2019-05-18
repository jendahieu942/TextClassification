class Utils:
    def __init__(self):
        self.list = []
        self.file = open('../data/vietnamese-stopwords.txt', 'r')

    def __del__(self):
        self.file.close()

    def load_stopwords(self):
        word = self.file.readline()
        while word:
            self.list.append(word.replace('\n', '').replace(' ', '_'))
            word = self.file.readline()
        return self.list


def remove_stopwords(texts, stopwords):
    texts_processed = []
    for doc in texts:
        spacy_doc = []
        for word in doc.split(' '):
            if word not in stopwords:
                spacy_doc.append(word)
        texts_processed.append(spacy_doc)
    return texts_processed
