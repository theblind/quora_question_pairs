import pandas as pd
import numpy as np
import os
import nltk
from collections import Counter
import data_processing


def question_match_tf_idf(data_question1, data_question2):
    """Calculate the match rate between two questions based on TF_IDF"""
    # Calculate IDF
    question_corpus = []
    question_corpus.extend(data_question1.tolist())
    question_corpus.extend(data_question2.tolist())
    text_collection = nltk.TextCollection(question_corpus)
    weights = {word: text_collection.idf(word)
               for word in text_collection.tokens}

    # Calculate the match rate
    result = []
    for question1, question2 in zip(data_question1, data_question2):
        result.append(match_rate_tf_idf(question1, question2, weights))
    return result


def match_rate_tf_idf(question1, question2, weights):
    """Calculate the match rate in words level"""
    question1_words = {}
    question2_words = {}
    for word in question1:
        question1_words[word] = 1
    for word in question2:
        question2_words[word] = 1
    if len(question1_words) == 0 or len(question2_words) == 0:
        return 0

    shared_weights = [weights.get(w) for w in question1_words.keys()
                      if w in question2_words]
    shared_weights.extend([weights.get(w) for w in question2_words.keys()
                           if w in question1_words])
    total_weights = [weights.get(w) for w in question1_words]
    total_weights.extend([weights.get(w) for w in question2_words])

    match_rate = np.sum(shared_weights) / np.sum(total_weights)
    return match_rate
