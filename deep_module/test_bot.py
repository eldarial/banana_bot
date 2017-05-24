from collections import Counter
import numpy as np
import sklearn
import pickle
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.callbacks import ModelCheckpoint


# Function to load parameters of word2vec
def load_word2vec_parameters(review_vocab_path, word2index_path):
    with open(review_vocab_path, 'rb') as input:
        review_vocab = pickle.load(input)
    with open(word2index_path, 'rb') as input:
        word2index = pickle.load(input)
    print("loaded word2vec parameters")
    return review_vocab, word2index


# convert string reviews to vectors
def encode_review(review, word2index, vocab):
    vocab_size = len(vocab)
    layer_0 = np.zeros((1, vocab_size))
    for word in review.split(" "):
        if word in word2index.keys():
            layer_0[0][word2index[word]] += 1
    return layer_0


# main code to test
def main():
    g = open('reviews.txt', 'r')  # What we know!
    reviews = list(map(lambda x: x[:-1],g.readlines()))
    g.close()

    g = open('labels.txt', 'r')  # What we WANT to know!
    labels = list(map(lambda x: x[:-1].upper(),g.readlines()))
    g.close()

    review_vocab, word2index = load_word2vec_parameters('train_vocab.pkl', 'train_word2index.pkl')
    vec = encode_review(reviews[0], word2index, review_vocab)

    # load model

    # make prediction


if __name__ == "__main__":
    main()

