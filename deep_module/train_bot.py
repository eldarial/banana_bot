from collections import Counter
import numpy as np
import sklearn

from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.preprocessing import sequence


# function to create word2index based on reviews in the dataset
def pre_process_data(reviews, labels, polarity_cutoff=0.1, min_count=10):
        positive_counts = Counter()
        negative_counts = Counter()
        total_counts = Counter()

        for i in range(len(reviews)):
            if labels[i] == 'POSITIVE':
                for word in reviews[i].split(" "):
                    positive_counts[word] += 1
                    total_counts[word] += 1
            else:
                for word in reviews[i].split(" "):
                    negative_counts[word] += 1
                    total_counts[word] += 1

        pos_neg_ratios = Counter()

        for term, cnt in list(total_counts.most_common()):
            if cnt >= 50:
                pos_neg_ratio = positive_counts[term] / float(negative_counts[term]+1)
                pos_neg_ratios[term] = pos_neg_ratio

        for word,ratio in pos_neg_ratios.most_common():
            if ratio > 1:
                pos_neg_ratios[word] = np.log(ratio)
            else:
                pos_neg_ratios[word] = -np.log((1 / (ratio + 0.01)))
        
        review_vocab = set()
        for review in reviews:
            for word in review.split(" "):
                if total_counts[word] > min_count:
                    if word in pos_neg_ratios.keys():
                        if(pos_neg_ratios[word] >= polarity_cutoff) or (pos_neg_ratios[word] <= -polarity_cutoff):
                            review_vocab.add(word)
                    else:
                        review_vocab.add(word)
        review_vocab = list(review_vocab)
        
        label_vocab = set()
        for label in labels:
            label_vocab.add(label)
        
        label_vocab = list(label_vocab)
        
        review_vocab_size = len(review_vocab)
        label_vocab_size = len(label_vocab)
        
        word2index = {}
        for i, word in enumerate(review_vocab):
            word2index[word] = i
        
        label2index = {}
        for i, label in enumerate(label_vocab):
            label2index[label] = i
        return review_vocab, label_vocab, word2index, label2index


# convert string reviews to vectors
def encode_review(review, word2index, vocab):
    vocab_size = len(vocab)
    layer_0 = np.zeros((1, vocab_size))
    for word in review.split(" "):
        if word in word2index.keys():
            layer_0[0][word2index[word]] += 1
    return layer_0


# generator for reviews and labels
def review_label_generator(reviews, labels, word2index, vocab, batch_size=4):
    x_batch = []
    y_batch = []
    num_samples = len(reviews)
    # while 1:
    for kl in range(1):
        sklearn.utils.shuffle(reviews, labels)
        for offset in range(0, int(np.floor(num_samples/batch_size)), batch_size): 
            batch_samples = reviews[offset:offset + batch_size]
            batch_labels = labels[offset:offset + batch_size]
            for each_sample, each_label in zip(batch_samples, batch_labels):
                x_batch.append(encode_review(each_sample, word2index, vocab))
                if each_label == 'POSITIVE':
                    y_batch.append(1)
                else:
                    y_batch.append(0)
            x_train = np.array(x_batch)
            y_train = np.array(y_batch)
    yield x_train, y_train


def main():
    # start code
    g = open('reviews.txt', 'r')  # What we know!
    reviews = list(map(lambda x: x[:-1],g.readlines()))
    g.close()

    g = open('labels.txt', 'r')  # What we WANT to know!
    labels = list(map(lambda x: x[:-1].upper(),g.readlines()))
    g.close()

    positive_counts = Counter()
    negative_counts = Counter()
    total_counts = Counter()

    for i in range(len(reviews)):
        if labels[i] == 'POSITIVE':
            for word in reviews[i].split(" "):
                positive_counts[word] += 1
                total_counts[word] += 1
        else:
            for word in reviews[i].split(" "):
                negative_counts[word] += 1
                total_counts[word] += 1

    print(len(reviews),len(labels))
    print('1asd', reviews[0], labels[0])
    print('1asd', reviews[10], labels[10])

    vocab = set(total_counts.keys())
    vocab_size = len(vocab)
    print(vocab_size)
    word2index = {}

    for i, word in enumerate(vocab):
        word2index[word] = i

    for k in range(4):
        l0 = encode_review(reviews[k], word2index, vocab)
        print('old', l0[0], np.sum(l0[0]), len(l0[0]))

    # pre-process text data
    review_vocab, label_vocab, reduced_word2index, label2index = pre_process_data(reviews, labels)
    print(len(review_vocab))
    for k in range(4):
        l0 = encode_review(reviews[k], reduced_word2index, review_vocab)
        print('new', l0[0], np.sum(l0[0]), len(l0[0]))

    # create the model
    model = Sequential()
    # model.add(Embedding(top_words, 32, input_length=max_words)) len(review_vocab)
    model.add(Conv1D(input_dim=len(review_vocab), filters=32, kernel_size=3, strides=1, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    # model.add(Flatten())
    model.add(Dense(250, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    # Fit the model
    # model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2, batch_size=128)
    # Final evaluation of the model
    # scores = model.evaluate(X_test, y_test, verbose=0)
    # print("Accuracy: %.2f%%" % (scores[1]*100))

if __name__ == "__main__":
    main()
