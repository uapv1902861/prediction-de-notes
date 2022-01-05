from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn import svm
import numpy as np
import pandas as pd
from nltk.stem.snowball import FrenchStemmer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm
import sys

"""

    Reduce polarity file to have less to process, based on train
    
"""

filename = "data/train_fixup.csv"

df = pd.read_csv( filename )
rowcount = int(len(df.index))#int(len(df.index)*0.8)
del df

polarity = pd.read_csv( "polarity-fr.txt", 
                       sep=";", 
                       comment="/", 
                       header=None,
                       usecols=[1,2,3,4],
                       names=["terme","positif","neutre","negatif"],
                       dtype={"terme": str, "positif": np.uint8, "neutre": np.uint8, "negatif": np.uint8}
                       )

polarity["terme"] = polarity["terme"].str.lower()
polarity = polarity.groupby('terme').mean().reset_index()
polarity.set_index(polarity.terme, inplace=True, drop=True)

# Done in parts as it takes a lot of ram
print("Row count:",rowcount)
range_increm = 100000
ranges = []
range_iter = 0
while( range_iter < rowcount ):
    if( range_iter + range_increm > rowcount ):
        ranges.append( (range_iter, rowcount) )
        break
    ranges.append( (range_iter, range_iter+range_increm) )
    range_iter+=range_increm
print("Cuts:",len(ranges))

open('vocab_processed.csv', "w").write("terme;positif;neutre;negatif")

for r in ranges:

    print("+ Processing:", r)
    df = pd.read_csv( filename ).iloc[r[0]:r[1],]
    
    df.replace(to_replace=[None], value='-', inplace=True)
    
    cv = CountVectorizer(  analyzer='word',
                           ngram_range=(1,4),
                           min_df=35,
                           stop_words=None,
                           dtype=np.uint8)
    
    print("# CountVectorizer Fit...")
    
    cv.fit(df["commentaire"])
    
    del df
    
    cols = cv.get_feature_names_out()
    
    
    print("# Polarity check append...")
    
    vocab = pd.DataFrame(columns = polarity.columns, data=None)
    for col in tqdm(cols):
        if( col in polarity.index ):
            vocab = pd.concat([vocab, polarity.loc[[col]]])
    
    print("# To CSV")
    
    vocab.to_csv('vocab_processed.csv', sep=";", index=False, mode='a', header=False)
    
    print("- done")
    
vocab = pd.read_csv( 'vocab_processed.csv', sep=";" )
vocab.drop_duplicates( inplace=True )
vocab.set_index(vocab.terme, inplace=True, drop=True)
vocab.to_csv('vocab_full_processed.csv', sep=";", index=False, mode='w')
