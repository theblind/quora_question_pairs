import os
import re

import nltk
import numpy as np
from gensim import corpora
from gensim.models import word2vec
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

EMBEDDING_DIMENSION = 100
MAX_LENGTH = 150


def tokenize(sentence):
    """Transfer a sentence to tokens"""
    try:
        tokens = nltk.word_tokenize(sentence)
        # stop_words = set(stopwords.words('english'))
        punctuation = ['.', ',', '"', "'", '?', '!', ':', ';',
                    '(', ')', '[', ']', '{', '}', '&', '!', '*', '@', '#', '$', '%']
        stop_words = punctuation
        porter = PorterStemmer()
        tokens = [porter.stem(i.lower())
                for i in tokens if i.lower() not in stop_words]
        return tokens
    except TypeError:
        return []


def tokenize_questions(questions):
    """Transfer questions to tokens"""
    questions = questions.map(lambda sentence: tokenize(sentence))
    return questions


def index_questions(questions, dictionary):
    """Transfer questions to indexes and padding them"""
    questions = questions.map(lambda sentence: dictionary.doc2idx(sentence))
    questions = pad_sequences(questions, maxlen=MAX_LENGTH, padding="post")
    return questions


def build_dictionary(question_corpus):
    """Build a dictionary for given question corpus"""
    dictionary = corpora.Dictionary(question_corpus)
    return dictionary


def train_word2vec(sentences, num_features=100, min_word_count=1, context=5):
    """Train word2vec Model"""
    dataset_path = "datasets"
    model_path = os.path.join(dataset_path, "word2vec_embedding.data")

    if os.path.exists(model_path):
        word2vec_embedding = word2vec.Word2Vec.load(model_path)
    else:
        word2vec_embedding = word2vec.Word2Vec(sentences,
                                               size=num_features,
                                               min_count=min_word_count,
                                               workers=-1,
                                               window=context)
        word2vec_embedding.init_sims(replace=True)
        word2vec_embedding.save(model_path)

    return word2vec_embedding


def load_glove(glove_path):
    """Load pre-trained Glove embedding"""
    word_embeddings = {}
    with open(glove_path, encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            embedding = np.asarray(values[1:], dtype='float32')
            word_embeddings[word] = embedding
    return word_embeddings


def get_embedding_matrix(word_embeddings, token2id):
    """Use word_embeddings to create embedding matrix"""
    embedding_matrix = np.zeros((len(token2id)+1, EMBEDDING_DIMENSION))
    for word, index in token2id.items():
        if word in word_embeddings:
            embedding_matrix[index] = word_embeddings[word]
        else:
            embedding_matrix[index] = np.random.uniform(-0.25, 0.25,
                                                        size=EMBEDDING_DIMENSION)
    return embedding_matrix
