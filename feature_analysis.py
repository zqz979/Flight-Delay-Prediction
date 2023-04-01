import numpy as np
import math
import utils
from sklearn.feature_selection import SelectKBest

AMNT_DROPPED=0.99

def ordinal_best(df):
    features,labels=utils.feature_label_split(df)
    features=utils.preproc_features_df(features,ordinal=True)
    labels=utils.preproc_labels_df(labels)
    skb=SelectKBest(k="all").set_output(transform="pandas")
    kdf=skb.fit_transform(features,labels)
    print("Ordinal:")
    for s,n in zip(skb.scores_,skb.feature_names_in_):
        print(n,s)

def one_hot_best(df):
    drop_inds = np.random.choice(df.index, math.floor(len(df)*AMNT_DROPPED), replace=False)
    df = df.drop(drop_inds)
    features,labels=utils.feature_label_split(df)
    features=utils.preproc_features_df(features)
    labels=utils.preproc_labels_df(labels)
    skb=SelectKBest(k="all").set_output(transform="pandas")
    kdf=skb.fit_transform(features,labels)
    print("One-hot:")
    for s,n in zip(skb.scores_,skb.feature_names_in_):
        print(n,s)

def main():
    df=utils.load_data(separate=False)
    ordinal_best(df)
    one_hot_best(df)

if __name__ == "__main__":
    main()