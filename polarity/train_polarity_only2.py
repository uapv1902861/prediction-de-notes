import pandas as pd
from sklearn import svm
import numpy as np

"""
    Predict using Polarity [Positif/Neutre, Negatif/Neutre]
    
    Requires test_polarized.csv and train_polarized.csv from polarize_data.py
    
"""

line_count = 20000

x_train = pd.read_csv("train_polarized.csv", sep=";").head(line_count)
x_train.drop("review_id", axis=1, inplace=True)
y_train = pd.read_csv("../data/train_fixup.csv", usecols=["note"])["note"].head(line_count)


"""
def normalize(df):
    for i,r in df.iterrows():
        m = r["positif"] + r["neutre"] + r["negatif"]
        if(m == 0):
            return
        r["positif"] = r["positif"] / m
        r["neutre"] = r["neutre"] / m
        r["negatif"] = r["negatif"] / m
"""

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

print("Process columns")

x_test = pd.read_csv("test_polarized.csv", sep=';')
x_test_id = x_test['review_id']
x_test.drop('review_id', axis=1, inplace=True)

x_train.fillna(0.01, inplace=True)
x_test.fillna(0.01, inplace=True)

x_train = posdivneg(x_train)
x_test = posdivneg(x_test)

print("Train")
svc = svm.SVC(gamma='scale')
svc.fit(x_train,y_train)

print("Predict")
results = svc.predict(x_test)

print("resultat")
f = open("resultat_polarity_only2.txt", "w", encoding='ascii')
for rvw, pred in zip(x_test_id, results):
    tmp = rvw + " " + pred + "\n"
    f.write(tmp)

f.close()
