# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import pandas as pd
import regex as re
import tensorflow as tf
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Embedding, Flatten, Dropout
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

def get_data( path ):
    file = open( path, "r", encoding="utf-8")
    lines = file.read()
    file.close()

    # Fix '&' to '&amp;'
    regex = re.compile(r"&(?!amp;|lt;|gt;)")
    lines = regex.sub("&amp;", lines)

    # XML Strings -> Dataframe
    root = pd.read_xml(lines)
    

    return root

print("Récupération des données")
data = get_data('../data/train.xml' )

data_test = get_data('../data/test.xml')

#On remplace les commentaires vide dans nos corpus
data.replace(to_replace=[None], value='-', inplace=True)

data_test.replace(to_replace=[None], value='-', inplace=True)

#On découpe split ensuite nos données d'apprentissage
x_train, x_test, y_train, y_test = train_test_split( data['commentaire'], data['note'], test_size=0.025)

#Afin de ne récupérer que 10000 commentaires sur l'ensemble des données
x_train, x_test, y_train, y_test = train_test_split(x_test, y_test, test_size=0.3993)

print("Tokenizer:")
#On initialise les tokenizers de keras
tokenizer = Tokenizer()
test_tokenizer = Tokenizer()

full_reviews = x_train.append(x_test)

print("Fit on texts")
#On fit l'ensemble des commentaires avec le tokenizer
tokenizer.fit_on_texts(full_reviews)
test_tokenizer.fit_on_texts(data_test['commentaire'])

#On récupère ensuite le taille maximale d'un commentaire
max_length = max([len(s.split()) for s in data_test['commentaire']])

#Ainsi que la nombre de tokens uniques retenus
vocab_size = len(test_tokenizer.word_index)+1

print("Text to sequences")
#On transforme ensuite le texte en vecteurs numérique 
x_train_tokens = tokenizer.texts_to_sequences(x_train)
x_test_tokens = tokenizer.texts_to_sequences(x_test)

print("Pad sequences")
#Et on leur applique une taille maximale égale à la taille de max d'un commentaire.
x_train_pad = pad_sequences(x_train_tokens, maxlen=max_length, padding='post')
x_test_pad = pad_sequences(x_test_tokens, maxlen=max_length, padding='post')

print("LabelEncoder")
#On applique un LabelEncoder sur les notes du corpus d'apprentissage 
le = preprocessing.LabelEncoder()

y_train = le.fit_transform(y_train)
y_test = le.fit_transform(y_test)

#On transforme ensuite les valeurs encodées en valeurs difinissant un classes (une note)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

#On instantie le chemin du fichier du meilleur modèle
checkpoint_filepath='model_best.h5'

print("Model:")
#On prend une dimenssion d'embedding de 100
embedding_dim = 100

#On instancie ensuite le modèle de notre réseau de neurones
model = Sequential()		
#La couche d'embedding
model.add(Embedding(vocab_size, embedding_dim, input_length=max_length))
#Une couche flatten afin de ramener nos données à des vecteurs à deux dimmensions
model.add(Flatten())
#Notre couche de réseau de neurones
model.add(Dense(128, activation='relu'))
#Une couche de sortie de taille 10 car on cherche à prédire 10 classes
model.add(Dense(10, activation='softmax'))

#On compile notre modèle
model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(), metrics=[tf.keras.metrics.CategoricalAccuracy()])

model.summary()

#On instancie le callback permettant de sauvegarder le meilleurs modèle lors de l'apprentissage
my_callback = [
	tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=False, monitor='val_categorical_accuracy', save_best_only=True)
]

print("Apprentissage:")
#On lance ensuite l'apprentissage
hist = model.fit(x_train_pad, y_train, epochs=10, validation_data=(x_test_pad, y_test), verbose=1, callbacks=[my_callback])

