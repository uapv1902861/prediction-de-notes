# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import pandas as pd
import regex as re

import string
import gensim
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize

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

print("Récupération des données...")
data = get_data('../data/train.xml' )

#on récupère les stop-words
stop_words = open("stopwords-fr.txt", 'r', encoding="UTF-8").read().splitlines()

#On remplace les commentaires vide dans nos corpus
data.replace(to_replace=[None], value='-', inplace=True)

comments = data['commentaire'].values.tolist()
notes = data['note']

print("Split des données de train")

x_train, x_test, y_train, y_test = train_test_split(comments, notes, test_size=0.025)

x_train, x_test, y_train, y_test = train_test_split(x_test, y_test, test_size=0.3993)

commentaires = list()

print("Création des tokens")

for line in x_train:
    tokens = word_tokenize(line)
    tokens = [w.lower() for w in tokens] #passage en lower case
    table = str.maketrans('', '', string.punctuation) 
    stripped = [w.translate(table) for w in tokens] #retire la ponctuation
    words = [word for word in stripped if word.isalpha()] #retire les tokens non-alphabetic
    words = [w for w in words if not w in stop_words] #retire les mots fesant partie des stopwords
    commentaires.append(words)

print("Train du model word2vec")
model = gensim.models.Word2Vec(sentences=(commentaires), vector_size=100, window=5, workers=2, min_count=1)

print("Taille du vocabulaire:", len(model.wv))

#Enfin, on sauvegarde le modèle de notre word2vec
file = 'word2vec.txt'

model.wv.save_word2vec_format(file, binary=False)



