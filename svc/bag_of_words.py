# -*- coding: utf-8 -*-
import pandas as pd
import regex as re
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.model_selection import train_test_split

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

print("Récupération des corpus")
data = get_data('../data/train.xml' )

data_test = get_data('../data/test.xml')

print("Traitement du text")
#On remplace les commentaires vide dans nos corpus
data.replace(to_replace=[None], value='-', inplace=True)

data_test.replace(to_replace=[None], value='-', inplace=True)

#On découpe split ensuite nos données d'apprentissage
x_train, x_test, y_train, y_test = train_test_split(data['commentaire'], data['note'], test_size=0.025)


#Afin de ne récupérer que 10000 commentaires sur l'ensemble des données
x_train, x_test, y_train, y_test = train_test_split(x_test, y_test, test_size=0.3993)

print("Début du BOW apprentissage")

#On initialise le vectorizer qui va produire le bag of words
cv = CountVectorizer(analyzer='word', lowercase=True, ngram_range=(1,1), min_df=25, dtype=np.uint8)

#On applique ensuite la méthode
comm_counts = cv.fit_transform(x_train)

#Et récupère la sortie du vectorizer, le bag of words
df_words = pd.DataFrame(comm_counts.toarray(), columns=cv.get_feature_names_out())

svc = svm.SVC(gamma='scale')

print("Début du BOW de test")
comments = data_test['commentaire']

#On applique le même pré-traitement aux données de test
cv = CountVectorizer(vocabulary=cv.vocabulary_, analyzer='word', lowercase=True, ngram_range=(1,1), min_df=25, dtype=np.uint8)

comm_counts = cv.fit_transform(comments)

df_words_test = pd.DataFrame(comm_counts.toarray(), columns=cv.get_feature_names_out())

print("SVM fit")

#On commence l'apprentissage avec les données de train
svc.fit(df_words, y_train)

print("SVM predict")

#On lance ensuite la prédiction des notes
results = svc.predict(df_words_test)
  
print("Resultat")
#Enfin, on inscrit les résultats dans un fichier text
f = open("resultat.txt", "w", encoding='ascii')
for rvw, pred in tqdm(zip(data_test["review_id"], results)):
    tmp = rvw + " " + pred + "\n"
    f.write(tmp)

f.close()
