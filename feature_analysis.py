import numpy as np
import math
import utils
from sklearn.feature_selection import SelectKBest

AMNT_DROPPED=0.9

def main():
    df=utils.load_data(separate=False)
    drop_inds = np.random.choice(df.index, math.floor(len(df)*AMNT_DROPPED), replace=False)
    df = df.drop(drop_inds)
    features,labels=utils.feature_label_split(df)
    features=utils.preproc_features_df(features)
    labels=utils.preproc_labels_df(labels)
    skb=SelectKBest(k=10).set_output(transform="pandas")
    kdf=skb.fit_transform(features,labels)
    print(kdf.columns)

if __name__ == "__main__":
    main()