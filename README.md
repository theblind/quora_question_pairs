# Quora Question Pairs

This is a project for Kaggle Competition [Quora Question Pairs](https://www.kaggle.com/c/quora-question-pairs).
The competition is trying to identify the similarity between two questions. In Quora, there are lots of questions have same intention for specific topic. If we can merge those same intention questions to single question, it will greatly improve user experience.

We use a seq2seq model to address this problem:

* The input questions will be embedded by pre-trained Glove vectors, and feed into Bi-LSTM to get encoding state.
* After that, we create an intention vector between given two questions, and concatennate it and encoding state into another LSTM to get decoder state.
* At last, we use a fully connected layer and sigmoid activation to calculate a probility for the similarity between these two questions.

There are a lot of works we can do to improve the result, we can add the following features:

* Use doc2vec to encode the question and calculte the similarity.
* Use edit distance as an extra feature
* Ensemble the different learners by features
