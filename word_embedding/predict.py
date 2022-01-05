# -*- coding: utf-8 -*-
import pandas as pd
import regex as re
import numpy as np
from tensorflow.keras.models import load_model
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

#On récupère le chemin de notre meilleur modèle: model_best.h5 ou w2v_model_best.h5
checkpoint_filepath='w2v_model_best.h5'

data_test = get_data('../data/test.xml')

#On remplace les commentaires vide dans le corpus de test
data_test.replace(to_replace=[None], value='-', inplace=True)

comments = data_test['commentaire']

print("Tokenizer:")
#On instancie notre tokenizer
tokenizer = Tokenizer()

print("Fit on texts")
#On fit les commentaires
tokenizer.fit_on_texts(comments)

#puis on récupère la taille maximale pour un commentaire
max_length = max([len(s.split()) for s in comments])

#ainsi que le nombre de token dans notre vocabulaire
vocab_size = len(tokenizer.word_index)+1

print("Text to sequences")
#On transforme les commentaires tokenisés en vecteurs numériques
comments_tokens = tokenizer.texts_to_sequences(comments)

print("Pad sequences")
#Puis l'on ramène ses vecteurs a une taille fixe égale a la taille max d'un commentaire
comments_pad = pad_sequences(comments_tokens, maxlen=max_length, padding='post')

print("Load model")
#On charge le meilleur modèle trouvé lors de l'apprendtissage
model = load_model(checkpoint_filepath)

print("Début prédiction")
#On lance ensuite la prédiction
prediction = model.predict(comments_pad)

classes=np.argmax(prediction, axis=1)

print("Resultat")
print(classes)

#Efin, on enregiste le résultat de la prédiction dans un fichier texte
labels = {0:"0,5", 1:"1,0", 2:"1,5", 3:"2,0", 4:"2,5", 5:"3,0", 6:"3,5", 7:"4,0", 8:"4,5", 9:"5,0"}

notes_pred = []

for n in classes:
    notes_pred.append(labels[n])

f = open("resultat.txt", "w", encoding='ascii')
for rvw, pred in zip(data_test["review_id"], notes_pred):
    tmp = rvw + " " + pred + "\n"
    f.write(tmp)

f.close()