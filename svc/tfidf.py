# -*- coding: utf-8 -*-
import pandas as pd
import regex as re
import numpy as np
from tqdm import tqdm
from nltk.stem.snowball import FrenchStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
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


def stemmed_words(doc):
    return (stemmer.stem(w) for w in analyzer(doc))

print("Récupération des corpus")
data = get_data('../data/train.xml' )

data_test = get_data('../data/test.xml')
#On initialise le stemmer
stemmer = FrenchStemmer()

#On récupère les stop-words
stop_words = open("stopwords-fr.txt", 'r', encoding="UTF-8").read().splitlines()

print("Traitement du text")
#On remplace les commentaires vide dans nos corpus
data.replace(to_replace=[None], value='-', inplace=True)

data_test.replace(to_replace=[None], value='-', inplace=True)

#On découpe split ensuite nos données d'apprentissage
x_train, x_test, y_train, y_test = train_test_split(data['commentaire'], data['note'], test_size=0.025)

#Afin de ne récupérer que 10000 commentaires sur l'ensemble des données
x_train, x_test, y_train, y_test = train_test_split(x_test, y_test, test_size=0.3993)

print("Début du TF-IDF apprentissage")
#On initialise l'analyser du vectorizer afin de pouvoir utiliser une fonction personnalisée
analyzer = TfidfVectorizer().build_analyzer()

#On initialise ensuite le vectorizer
vect = TfidfVectorizer(analyzer='word', lowercase=True, ngram_range=(2,2), min_df=10, stop_words=stop_words, dtype=np.uint8)

#On applique la méthode de TF-IDF
tfidf_comms = vect.fit_transform(x_train)

#On récupère les vecteurs de poids calculés par le vectorizer
df_words = pd.DataFrame(tfidf_comms.toarray(), columns=vect.get_feature_names_out())

svc = svm.SVC(gamma='scale')

print("Début du TF-IDF de test")
comments = data_test['commentaire']

#On applique le même pré-traitement aux données de test
vect = TfidfVectorizer(vocabulary=vect.vocabulary_, analyzer='word', lowercase=True, ngram_range=(2,2), min_df=10, stop_words=stop_words, dtype=np.uint8)

comm_counts = vect.fit_transform(comments)

df_words_test = pd.DataFrame(comm_counts.toarray(), columns=vect.get_feature_names_out())

print("Apprentissage du SVC")

#On commence l'apprentissage avec les données de train
svc.fit(df_words, y_train)

print("Préduiction du SVC")

#On lance ensuite la prédiction des notes
results = svc.predict(df_words_test)
  
print("Inscription du résultat")

#Enfin, on inscrit les résultats dans un fichier text
f = open("resultat.txt", "w", encoding='ascii')
for rvw, pred in tqdm(zip(data_test["review_id"], results)):
    tmp = rvw + " " + pred + "\n"
    f.write(tmp)
f.close()
