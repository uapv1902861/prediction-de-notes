import numpy as np
import pandas as pd
from threading import Thread


required_columns = [ "review_id", "note", "commentaire" ]

path = "../data/"
file = "train"


# drop the columns we don't need
def drop_main_columns( df ):
    print("Droping columns...")
    for col in df.columns:
        if( col not in required_columns ):
            df = df.drop( col, axis=1 )
    print("- Done.\n")
    return df

def format_main_dataframe( df ):
    print("Converting data to be usable...")
    df.replace(to_replace=[None], value='-', inplace=True)
#    df['note'] = df['note'].str.replace(',', '.').astype(float)
    print("- Done.\n")
    return df

polarity = pd.read_csv( "vocab_full_reduced.csv", 
                       sep=";", 
                       # comment="/", 
                       # header=None,
                       # usecols=[1,2,3,4],
                       # names=["terme","positif","neutre","negatif"],
                       dtype={"terme": str, "positif": np.float64, "neutre": np.float64, "negatif": np.float64}
                       )
polarity.set_index(polarity.terme, inplace=True, drop=True)
polarity["terme"] = polarity["terme"].astype(str)

df = pd.read_csv(path+file+"_fixup.csv")
#df = pd.read_xml("data/"+file+".xml")

df = format_main_dataframe( df )
df = drop_main_columns( df )
df.set_index(df.review_id, inplace=True, drop=True)

"""
cv = CountVectorizer(  vocabulary=polarity["terme"],
                       analyzer='word',
                       min_df=35,
                       stop_words=None,
                       )

df_cv = cv.fit_transform( df["commentaire"].astype(str) )
"""

df["positif"] = 0
df["neutre"] = 0
df["negatif"] = 0

polarity_dict = polarity.to_dict(orient='index')

d1,d2,d3,d4 = np.split(df, [
    int(len(df)*0.25),
    int(len(df)*0.5),
    int(len(df)*0.75)
    ])

del df

d1 = d1.to_dict(orient='index')
d2 = d2.to_dict(orient='index')
d3 = d3.to_dict(orient='index')
d4 = d4.to_dict(orient='index')

# for all terms -> assign the score if found in the comment
def process_vocab(df_dict):
    for k in df_dict:
        for term_k,term_v in polarity_dict.items():
            if( term_k in df_dict[k]["commentaire"] ):
                df_dict[k]["positif"] += term_v["positif"]
                df_dict[k]["negatif"] += term_v["negatif"]
                df_dict[k]["neutre"] += term_v["neutre"]
                    
df_parts = []
df_parts.append(d1)
del d1
df_parts.append(d2)
del d2
df_parts.append(d3)
del d3
df_parts.append(d4)
del d4

threads = []

print("Processing")

for d in df_parts:
    p = Thread(target=process_vocab, args=[d])
    p.start()
    threads.append(p)
    
for t in threads:
    t.join()
    
print("Exporting")

open(file+'_polarized.csv', "w").write("review_id;positif;neutre;negatif\n")
for d in range(0,len(df_parts)):
    df_parts[d] = pd.DataFrame.from_dict(df_parts[d], orient='index', columns=["positif", "neutre", "negatif"])
    df_parts[d].to_csv(file+'_polarized.csv', sep=";", index=True, mode='a', header=False)
