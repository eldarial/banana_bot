from collections import Counter
import numpy as np
import sklearn
import pickle

from keras.models import load_model


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
    output_encoded = []
    for word in review.split(" "):
        if word in word2index.keys():
            layer_0[0][word2index[word]] += 1
    output_encoded.append(np.transpose(np.array(layer_0)))
    return np.array(output_encoded)


# main code to test
def main():
    g = open('reviews.txt', 'r')  # What we know!
    reviews = list(map(lambda x: x[:-1],g.readlines()))
    g.close()

    g = open('labels.txt', 'r')  # What we WANT to know!
    labels = list(map(lambda x: x[:-1].upper(),g.readlines()))
    g.close()

    index_sample = 0
    review_vocab, word2index = load_word2vec_parameters('train_vocab.pkl', 'train_word2index.pkl')
    vec = encode_review(reviews[index_sample], word2index, review_vocab)
    print(labels[index_sample], reviews[index_sample])

    # load model
    model_bot = load_model('model_bot.h5')

    # make prediction
    prediction = model_bot.predict(vec)
    if 1 == np.argmax(prediction[0]):
        message = 'your review is positive'
    else:
        message = 'your review is negative'
    print(message)


if __name__ == "__main__":
    main()

