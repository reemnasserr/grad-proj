import tensorflow as tf
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Embedding, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import json


data_directory_path = 'Data/Transcript/Segmented/'
filenames = os.listdir(data_directory_path)
labels = pd.read_csv('labels.csv')
glove_path = 'glove.840B.300d.txt'
checkpoint_path= 'D:\DarkSide\Graduation project'

data = []
targets = []

# loading glove word embeddings 

def pretrained_embeddings(file_path, EMBEDDING_DIM, VOCAB_SIZE, word2idx):  
    word2vec = {}
    with open(os.path.join(file_path),  errors='ignore', encoding='utf8') as f:
        for line in f:
            values = line.split(' ')
            word = values[0]
            vec = np.asarray(values[1:], dtype='float32')
            word2vec[word] = vec
    
    num_words = VOCAB_SIZE
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in word2idx.items():
      if i < VOCAB_SIZE:
          embedding_vector = word2vec.get(word)
          if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

# load data 

for filename in filenames:
    lines = []
    with open(data_directory_path + filename, 'r') as f:
        lines = f.readlines()
    
    for index, line in enumerate(lines):
        lines[index] = line.split('_')[2]
        lines[index] = lines[index].strip()
    
    data = data + lines
    filename_updated = filename.split('.')[0]
    
    current_labels = labels[labels['video_name'] == filename_updated]
    targets = targets + list(current_labels['label'].values)

targets = np.array(targets) 

# preprocess the data 

tokenizer = Tokenizer(num_words= 3150, filters= '')
tokenizer.fit_on_texts(data)
seq = tokenizer.texts_to_sequences(data)
seq_padded = pad_sequences(seq, padding= 'post')
tokenizer_config = tokenizer.get_config()
word_index = json.loads(tokenizer_config['word_index'])

embedding_matrix = pretrained_embeddings(glove_path, 300, 3150, word_index)


model = Sequential([
        Embedding(input_dim= 3150, output_dim= 300, mask_zero= True, 
                  embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix),
                  trainable=False),
        LSTM(64, return_sequences=True),
        LSTM(64),
        Dropout(.4),
        Dense(32, activation= 'relu'),
        Dropout(.4),
        Dense(32, activation= 'relu'),
        Dense(1, activation= 'sigmoid')
    ])
    
model.summary()  
    
model.compile(optimizer= Adam(0.0001), loss= 'binary_crossentropy', metrics= ['accuracy'])
save_waights = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True,verbose=1)

history = model.fit(x= seq_padded, y= targets, batch_size= 32, epochs= 20, validation_split= 0.15,
                    callbacks=[save_waights])


    
def plot_history(history):
    fig, axs = plt.subplots(2)

    # create accuracy sublpot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error sublpot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()

plot_history(history)
    
    
    
    
    