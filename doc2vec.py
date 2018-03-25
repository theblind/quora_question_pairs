import os

import gensim
import numpy as np
from scipy import spatial

import data_processing


def train_doc2vec(documents, m_iter=100, m_min_count=2, m_size=100, m_window=5):
    """Train doc2vec Model"""
    dataset_path = "datasets"
    model_path = os.path.join(dataset_path, "doc2vec_embedding.data")

    if os.path.exists(model_path):
        model = gensim.models.doc2vec.Doc2Vec.load(model_path)
    else:
        model = gensim.models.Doc2Vec(
            documents, vector_size=m_size, window=m_window, min_count=m_min_count, iter=m_iter)
        model.save(model_path)
    return model


def question_match_doc2vec(question1, question2, model):
    """Calculate the match rate between two questions based on doc2vec"""
    question1_vector = model.infer_vector(question1)
    question2_vector = model.infer_vector(question2)
    match_rate = 1 - spatial.distance.cosine(question1_vector, question2_vector)
    return match_rate
