import pandas as pd
from typing import Any
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Dense
from keras.models import Sequential
from nltk import tokenize
from tensorflow.keras.utils import to_categorical
import re
import numpy as np
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer
import nltk
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from main import checkAccuracy, findAccuracyPercentage
nltk.download('punkt')


lines = []
data = ""
vocabulary_size = Any
seq_len = Any
train_inputs = Any
train_targets = Any
tokenizer = Any

classifier_model = Any


def open_file():
    global lines
    file = open("lang-8-english-1.0.txt", "r", encoding="utf8")
    for i in file:
        lines.append(i)
    print("The First Line: ", lines[0])
    print("The Last Line: ", lines[-1])


def join_lines():
    global lines, data
    for i in lines:
        data = ' '. join(lines)


def preprocess():
    global vocabulary_size, seq_len, train_inputs, train_targets, tokenizer
    cleaned = re.sub(r'\W+', ' ', data).lower()
    tokens = word_tokenize(cleaned)
    train_len = 4
    text_sequences = []

    for i in range(train_len, len(tokens)):
        seq = tokens[i-train_len:i]
        text_sequences.append(seq)

    sequences = {}
    count = 1

    for i in range(len(tokens)):
        if tokens[i] not in sequences:
            sequences[tokens[i]] = count
            count += 1

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text_sequences)
    sequences = tokenizer.texts_to_sequences(text_sequences)

    # vocabulary size increased by 1 for the cause of padding
    vocabulary_size = len(tokenizer.word_counts)+1
    n_sequences = np.empty([len(sequences), train_len], dtype='int32')

    for i in range(len(sequences)):
        n_sequences[i] = sequences[i]

    train_inputs = n_sequences[:, :-1]
    train_targets = n_sequences[:, -1]
    train_targets = to_categorical(train_targets, num_classes=vocabulary_size)
    seq_len = train_inputs.shape[1]


def train_model():
    model = Sequential()
    model.add(Embedding(vocabulary_size, seq_len, input_length=seq_len))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(vocabulary_size, activation='softmax'))

    # compiling the network
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    model.fit(train_inputs, train_targets, epochs=50, verbose=1)
    model.save('uni-directional')


def load_model():
    global classifier_model
    classifier_model = tf.keras.models.load_model('uni-directional')
    # classifier_model.summary()


def predict_item(first, second):
    word = first+" "+second
    input_text = word.strip().lower()
    encoded_text = tokenizer.texts_to_sequences([input_text])[0]
    pad_encoded = pad_sequences(
        [encoded_text], maxlen=seq_len, truncating='pre')
    # print(encoded_text, pad_encoded)
    pred_word = ""
    for i in (classifier_model.predict(pad_encoded)[0]).argsort()[-1:][::-1]:
        pred_word = tokenizer.index_word[i]
    return pred_word


def predict_and_accuracy():
    print("Predicting....")


def csv_reader():
    dataframe = pd.read_csv('prepositions_missing.csv')
    return dataframe


def tokenizeSentence(sentence):
    tokens = sentence.split(" ")
    return tokens


def rearrangePrepositions(preps):
    prep_sets = preps.split(";")
    preps_list = []
    for prep in prep_sets:
        prep_split = prep.split(",")
        preps_list.append([int(prep_split[0]), prep_split[1]])

    return preps_list


def arrangeSentence(tokens):
    sentence = ""
    for token in tokens:
        sentence += token+" "

    return sentence


def start_unidirectional(uni_directional_sentences):
    open_file()
    join_lines()
    preprocess()
    # # train_model()
    load_model()
    # print("haaaaaaaaaaaaaaaaaaaaaaai")
    # print("Predicting  : ", predict_item("I", "go"))
    response = []

    data = uni_directional_sentences

    for i in range(0, len(uni_directional_sentences)):
        index, sentence, prepositions = uni_directional_sentences[i]
        # print()
        # print("Actual sentence  :", sentence)
        sentence_tokens = tokenizeSentence(sentence)
        # print(sentence_tokens)
        rearranged_prepositions = rearrangePrepositions(prepositions)
        predicted_senteces = []
        # print(rearranged_prepositions)
        unidirectional_accuracy_state = []
        for preps in rearranged_prepositions:
            prep_index = preps[0]
            # print(index)
            actual_prep = preps[1]
            predicted_prep = predict_item(
                sentence_tokens[prep_index-3], sentence_tokens[prep_index-2])
            sentence_tokens[prep_index-1] = predicted_prep
            predicted_sentence = arrangeSentence(sentence_tokens)
            # print("Predicted sentence :", predicted_sentence)

            predicted_senteces.append(predicted_sentence)
            unidirectional_accuracy_state.append(
                checkAccuracy(predicted_sentence, sentence))
            sentence_tokens = tokenizeSentence(sentence)
        prediction_accuracy = findAccuracyPercentage(
            len(rearranged_prepositions), unidirectional_accuracy_state)
        response.append([uni_directional_sentences[i][0],
                        sentence, predicted_senteces,prediction_accuracy])

        print("PREDICTION ACCURACY OF SENTENCE = ", prediction_accuracy)
    return response

# start()
