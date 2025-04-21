from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec

def tfidf_vectorize(corpus):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    return tfidf_matrix, vectorizer

def word_embedding_vectorize(corpus):
    tokenized_corpus = [text.split() for text in corpus]
    model = Word2Vec(tokenized_corpus, vector_size=100, window=5, min_count=1, workers=4)
    word_vectors = {word: model.wv[word] for word in model.wv.index_to_key}
    return word_vectors