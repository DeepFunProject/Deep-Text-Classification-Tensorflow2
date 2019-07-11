import numpy as np
import re
import itertools
from collections import Counter


positiveDataAddress = './data/rt-polarity.pos'
negativeDataAddress = './data/rt-polarity.neg'

def clean_str(string):
    """
    Tokenization/string cleaning for datasets.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()
  

def load_data_and_labels():
    """
    Loads polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from file
    positive_samples = list(open(positiveDataAddress, 'r', encoding='latin-1').readlines())
    positive_samples = [s.strip() for s in positive_samples]
    negative_samples = list(open(negativeDataAddress, 'r', encoding='latin-1').readlines())
    negative_samples = [s.strip() for s in negative_samples]
    
    # Split by words
    x_text = positive_samples + negative_samples
    x_text = [clean_str(sent) for sent in x_text]
    x_text = [s.split(" ") for s in x_text]
    
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_samples]
    negative_labels = [[1, 0] for _ in negative_samples]
    
    y = np.concatenate([positive_labels, negative_labels], 0)
    
    return [x_text, y]
    
def pad_sentences(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    print(sequence_length)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences
  
def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    vocabulary_inv = list(sorted(vocabulary_inv))
    
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]
  
def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentences and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]

  
def load_data():
    """
    Loads and preprocessed data for the dataset.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    sentences, labels = load_data_and_labels()
    sentences_padded = pad_sentences(sentences)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)
    return [x, y, vocabulary, vocabulary_inv]

#************************************************** model **************************************************************
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint

print('Load Data...')
x, y, vocabulary, vocabulary_inv = load_data()

x_shape = x.shape

TEST_SPLIT=0.1
VALIDATION_SPLIT=0.2

MAX_SEQUENCE_LENGTH = x_shape[1] # 56
vocabulary_size = len(vocabulary_inv) # 18765
EMBEDDING_DIM = 300
filter_sizes = [3,4,5]
num_filters = 64
drop = 0.2
n_epochs = 100
BATCH_SIZE = 30

print('The number of sentences is :', x_shape[0], ', The lenght of sentences is :', x_shape[1])
print('We have {} words in our sentences'.format(vocabulary_size))

# split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=TEST_SPLIT,
                                                        random_state=42)

    # split training data into train and validation
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=VALIDATION_SPLIT,
                                                      random_state=1)

    n_train_samples = X_train.shape[0]
    n_val_samples = X_val.shape[0]
    n_test_samples = X_test.shape[0]

    print('We have %d TRAINING samples' % n_train_samples)
    print('We have %d VALIDATION samples' % n_val_samples)
    print('We have %d TEST samples' % n_test_samples)

inputs = Input(shape=(MAX_SEQUENCE_LENGTH, ), dtype='int32')
embedding = Embedding(input_dim=vocabulary_size, output_dim=EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH)(inputs)
reshape = Reshape((MAX_SEQUENCE_LENGTH, EMBEDDING_DIM, 1))(embedding)

conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[0], EMBEDDING_DIM), strides=1, padding='valid', kernel_initializer='normal', activation='relu')(reshape)
bn_0 = BatchNormalization()(conv_0)
conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[1], EMBEDDING_DIM), strides=1, padding='valid', kernel_initializer='normal', activation='relu')(reshape)
bn_1 = BatchNormalization()(conv_1)
conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[2], EMBEDDING_DIM), strides=1, padding='valid', kernel_initializer='normal', activation='relu')(reshape)
bn_2 = BatchNormalization()(conv_2)

maxpool_0 = MaxPool2D(pool_size=(MAX_SEQUENCE_LENGTH - filter_sizes[0] + 1, 1), padding='valid')(bn_0)
maxpool_1 = MaxPool2D(pool_size=(MAX_SEQUENCE_LENGTH - filter_sizes[1] + 1, 1), padding='valid')(bn_1)
maxpool_2 = MaxPool2D(pool_size=(MAX_SEQUENCE_LENGTH - filter_sizes[2] + 1, 1), padding='valid')(bn_2)

concatenated_tensor_1 = keras.layers.concatenate([maxpool_0, maxpool_1, maxpool_2])
flatten_1 = Flatten()(concatenated_tensor_1)
dense_1 = Dense(units=10, activation='tanh')(flatten_1)
dropout_1 = Dropout(drop)(dense_1)

lstm_0 = LSTM(64, return_sequences=True, activation='relu')(embedding)
bn_3 = BatchNormalization()(lstm_0)
lstm_1 = LSTM(32, return_sequences=False, activation='relu')(bn_3)
bn_4 = BatchNormalization()(lstm_1)
dense_2 = Dense(units=10, activation='tanh')(bn_4)
dropout_2 = Dropout(drop)(dense_2)

concatenated_tensor_2 = keras.layers.concatenate([dropout_1, dropout_2])
output = Dense(units=2, activation='softmax')(concatenated_tensor_2)

model = keras.Model(inputs=inputs, outputs=output)
early = keras.callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0.003,
                              patience=5,
                              verbose=1, mode='auto')

model.compile(optimizer=keras.optimizers.Adam(lr=0.001),
             loss=keras.losses.BinaryCrossentropy(),
             metrics=['accuracy'])
model.summary()

print("Traning Model...")
history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=n_epochs,  callbacks=[early,], validation_data=(X_val, y_val))

model.save('drive/My Drive/Text_Classification/model_one/model.h5')

import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'], loc='upper left')
plt.show()

res = model.evaluate(X_test, y_test)

predicted_labels = model.predict(X_test)
for i in range(predicted_labels.shape[0]):
  print(np.argmax(predicted_labels[i]), '\t', np.argmax(y_test[i]))

