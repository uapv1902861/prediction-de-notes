import pandas as pd
import regex as re
import numpy as np
from tqdm import tqdm
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.stem.snowball import FrenchStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import svm
from sklearn.model_selection import train_test_split

"""
    Predict using TF-IDF and Polarity
    
    - With stemmed words, lowercase, stopwords, 1-gram
    - [positif/neutre, negatif/neutre] 
    
    Requires test_polarized.csv and train_polarized.csv from polarize_data.py
    
"""

def get_data( path ):
    file = open( path, "r", encoding="utf-8")
    lines = file.read()
    file.close()

    # Fix '&' to '&amp;'
    regex = re.compile(r"&(?!amp;|lt;|gt;)")
    lines = regex.sub("&amp;", lines)

    # XML Strings -> Dataframe
    root = pd.read_xml(lines)
    
    #if path != 'donnees_appr_dev/test.xml':
    #    root['note'] = root['note'].str.replace(',', '.')

    return root

def stemmed_words(doc):
    return (stemmer.stem(w) for w in analyzer(doc))

data = get_data('../data/train.xml' )

data_test = get_data('../data/test.xml')

stemmer = FrenchStemmer()

stop_words = open("../stopwords-fr.txt", 'r', encoding="UTF-8").read().splitlines()

data.replace(to_replace=[None], value='-', inplace=True)

data_test.replace(to_replace=[None], value='-', inplace=True)

comments = data['commentaire']
notes = data['note']

x_train, x_test, y_train, y_test = train_test_split(comments, notes, test_size=0.025)

#del data

x_train, x_test, y_train, y_test = train_test_split(x_test, y_test, test_size=0.3993)


#cv = CountVectorizer(analyzer='word', lowercase=True, ngram_range=(1,1), min_df=25, stop_words=stop_words, dtype=np.uint8)

#comm_counts = cv.fit_transform(x_train)

analyzer = TfidfVectorizer().build_analyzer()

vect = TfidfVectorizer(analyzer=stemmed_words, lowercase=True, ngram_range=(1,1), min_df=25, stop_words=stop_words, dtype=np.int8)

tfidf_comms = vect.fit_transform(x_train)


df_words = pd.DataFrame(tfidf_comms.toarray(), columns=vect.get_feature_names_out())

del tfidf_comms

svc = svm.SVC(gamma='scale')

print("TFIDF")
#cv = CountVectorizer(vocabulary=cv.vocabulary_, analyzer='word', lowercase=True, ngram_range=(1,1), min_df=25, stop_words=stop_words, dtype=np.uint8)

#vect = TfidfVectorizer(vocabulary=vect.vocabulary_, analyzer=stemmed_words, lowercase=True, ngram_range=(1,1), min_df=25, stop_words=stop_words, dtype=np.uint8)
vect = TfidfVectorizer(analyzer='word', lowercase=True, ngram_range=(1,1), min_df=50, dtype=np.uint8)

comments = data_test['commentaire']

comm_counts = vect.fit_transform(comments)

del comments

df_words_test = pd.DataFrame(comm_counts.toarray(), columns=vect.get_feature_names_out())

del comm_counts


def posdivneg(df):
    newdf = pd.DataFrame.from_dict({"positif_ratio":[],"negatif_ratio":[]})
    
    for i,r in df.iterrows():
        p = 0.0
        n = 0.0
        # neutre can be 0
        if(r["neutre"] > 0.0):
            p = r["positif"] / r["neutre"]
            n = r["negatif"] / r["neutre"]
        else:
            p = r["positif"] / 0.01
            n = r["negatif"] / 0.01
    
        newdf = pd.concat([newdf,pd.DataFrame.from_dict({"positif_ratio":[p],"negatif_ratio":[n]})], ignore_index=True)
        
    return newdf

print("Process polarity columns and append (TRAIN)")

x_train_pol = pd.read_csv("train_polarized.csv", sep=";").head(len(df_words.index))
x_train_pol.drop("review_id", axis=1, inplace=True)

x_train_pol.fillna(0.0000, inplace=True)

train_mix = pd.concat([df_words,posdivneg(x_train_pol)], axis=1)

del x_train_pol, df_words

print("SVM fit")

svc.fit(train_mix, y_train)

del train_mix, y_train

print("Process polarity columns and append (TEST)")

x_test_pol = pd.read_csv("test_polarized.csv", sep=";").head(len(df_words_test.index))
x_test_pol.drop("review_id", axis=1, inplace=True)

x_test_pol.fillna(0.0000, inplace=True)

test_mix = pd.concat([df_words_test,posdivneg(x_test_pol)], axis=1)

del df_words_test, x_test_pol

print("SVM predict")

results = svc.predict(test_mix)
  
print("resultat")
f = open("resultat_mixed.txt", "w", encoding='ascii')
for rvw, pred in tqdm(zip(data_test["review_id"], results)):
    tmp = rvw + " " + pred + "\n"
    f.write(tmp)

f.close()
