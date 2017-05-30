from slackbot.bot import Bot, respond_to
import tensorflow as tf
import numpy as np
import logging
import re
import requests
import html
import slackbot_settings as settings
import pickle 

from keras.models import load_model
from keras import backend as K



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


@respond_to('(.*)', re.IGNORECASE)
def hi(message, phrase):
    review_vocab, word2index = load_word2vec_parameters('deep_module/train_vocab.pkl', 'deep_module/train_word2index.pkl')
    vec = encode_review(phrase, word2index, review_vocab)
    
    model_bot = load_model('deep_module/model_bot.h5')
    prediction = model_bot.predict(vec)
    print(prediction[0])
    if 1 == np.argmax(prediction[0]):
        message.reply('your review is positive')
        message.react('+1')
    else:
        message.reply('your review is negative')
        message.react('-1')
    K.clear_session()


@respond_to('I love plantains')
def love(message):
    message.reply('me too :heart:')
    

def main():
    logging.basicConfig(level=logging.DEBUG)
    bot = Bot()
    bot.run()


if __name__ == "__main__":
    main()
