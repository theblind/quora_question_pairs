import os

import keras
import numpy as np
import pandas as pd
from keras.layers import Concatenate, Dense, Dropout, Flatten, Merge, Reshape
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.model_selection import train_test_split

import data_processing

EMBEDDING_DIMENSION = 100
HIDDEN_DIMENSION = 100
BATCH_SIZE = 100
NUM_EPOCHES = 20
MAX_LENGTH = 150
DROPOUT_RATE = 0.25


def build_model(embedding_matrix, vocabulary_size):
    # Encoder question1
    question1_encoder = Sequential()
    question1_encoder.add(Embedding(input_dim=vocabulary_size,
                                    output_dim=EMBEDDING_DIMENSION,
                                    input_length=MAX_LENGTH,
                                    weights=[embedding_matrix],
                                    mask_zero=True))
    question1_encoder.add(LSTM(HIDDEN_DIMENSION, return_sequences=True))
    question1_encoder.add(Dropout(0.25))

    # Encoder question2
    question2_encoder = Sequential()
    question2_encoder.add(Embedding(input_dim=vocabulary_size,
                                    output_dim=EMBEDDING_DIMENSION,
                                    input_length=MAX_LENGTH,
                                    weights=[embedding_matrix],
                                    mask_zero=True))
    question2_encoder.add(LSTM(HIDDEN_DIMENSION, return_sequences=True))
    question2_encoder.add(Dropout(0.25))

    # Add attention between question1 and question2
    attention = Sequential()
    attention.add(
        Merge([question1_encoder, question2_encoder], mode="dot", dot_axes=[1, 1]))
    attention.add(Flatten())
    attention.add(Dense((MAX_LENGTH * HIDDEN_DIMENSION)))
    attention.add(Reshape((MAX_LENGTH, HIDDEN_DIMENSION)))

    # Add decoder layer
    model = Sequential()
    model.add(Merge([question1_encoder, question2_encoder,
                     attention], mode="concat", concat_axis=2))
    model.add(LSTM(HIDDEN_DIMENSION))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model


def main():
    # Read training and testing data
    dataset_path = "datasets"
    train_data_path = os.path.join(dataset_path, "train.csv")
    train_data = pd.read_csv(train_data_path).head(100)
    test_data_path = os.path.join(dataset_path, "test.csv")
    test_data = pd.read_csv(test_data_path).head(100)

    # Transfer questions to tokens
    train_data_question1 = data_processing.tokenize_questions(train_data.question1)
    train_data_question2 = data_processing.tokenize_questions(train_data.question2)
    test_data_question1 = data_processing.tokenize_questions(test_data.question1)
    test_data_question2 = data_processing.tokenize_questions(test_data.question2)

    # Build a dictionary for questions corpus
    question_corpus = []
    question_corpus.extend(train_data_question1.tolist())
    question_corpus.extend(train_data_question2.tolist())
    question_corpus.extend(test_data_question1.tolist())
    question_corpus.extend(test_data_question2.tolist())
    dictionary = data_processing.build_dictionary(question_corpus)
    vocabulary_size = len(dictionary.token2id)+1

    # Transfer questions to index representation
    train_data_question1 = data_processing.index_questions(train_data_question1, dictionary)
    train_data_question2 = data_processing.index_questions(train_data_question2, dictionary)
    test_data_question1 = data_processing.index_questions(test_data_question1, dictionary)
    test_data_question2 = data_processing.index_questions(test_data_question2, dictionary)

    # Create Embedding matrix
    glove_path = os.path.join(
        dataset_path, "glove.6B.{}d.txt".format(EMBEDDING_DIMENSION))
    word_embeddings = data_processing.load_glove(glove_path)
    embedding_matrix = data_processing.get_embedding_matrix(
        word_embeddings, dictionary.token2id)

    # Build the model and train
    model = build_model(embedding_matrix, vocabulary_size)
    model.fit([train_data_question1, train_data_question2], train_data.is_duplicate,
              batch_size=BATCH_SIZE,
              epochs=NUM_EPOCHES)

    # Predict the result for test data
    predicts = model.predict([test_data_question1, test_data_question2])
    
    # Save the result
    submittion = pd.DataFrame()
    submittion["test_id"] = test_data.test_id
    submittion["is_duplicate"] = predicts
    submittion.to_csv("my_submission.csv", index=False)


if __name__ == "__main__":
    main()
