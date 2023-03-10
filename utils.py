import pandas as pd
import numpy as np
import yfinance as yf
import os
import scipy
from sklearn.preprocessing import OneHotEncoder

CODE2STOCK={"UA":"UAL","AA":"AAL","AS":"ALK","B6":"JBLU","DL":"DAL","HA":"HA","NK":"SAVE","OO":"SKYW","WN":"LUV","G4":"ALGT"}
NO_STOCK=["EV","VX","9E","MQ","OH","YX","QX","F9","YV"]

def _clean_bts_data(input_dir_path="./raw_bts_data",output_path="./data/bts_data.csv"):
    df=None
    for f in os.listdir(input_dir_path):
        p = os.path.join(input_dir_path, f)
        if(type(df)==None):
            df=pd.read_csv(p)
        else:
            df=pd.concat([df,pd.read_csv(p)])
    df=df.dropna()
    for s in NO_STOCK:
        df=df[df["OP_UNIQUE_CARRIER"]!=s]
    df.to_csv(output_path,index=False)    

def _get_dow(input_path="./raw_stock_data/dow_data.csv"):
    df=pd.read_csv(input_path)
    df["Stock"]="DOW"
    df["YEAR"]=df.apply(lambda row: int(row["Date"].split("/")[2]),axis=1)
    df["MONTH"]=df.apply(lambda row: int(row["Date"].split("/")[0]),axis=1)
    df["DAY_OF_MONTH"]=df.apply(lambda row: int(row["Date"].split("/")[1]),axis=1)
    df=df.drop("Date",axis=1)
    return df

def _clean_stock_data(input_dow_path="./raw_stock_data/dow_data.csv",output_path="./data/stock_data.csv"):
    ranges=[("2017-12-01","2017-12-31"),
            ("2018-12-01","2018-12-31"),
            ("2019-12-01","2019-12-31"),
            ("2020-12-01","2020-12-31"),
            ("2021-12-01","2021-12-31"),
            ]
    stocks=list(CODE2STOCK.values())
    stocks.append("^IXIC")
    data=None
    for stock in stocks:
        for range in ranges:
            new = yf.download(stock, start=range[0], end=range[1])
            new["Stock"]=stock
            if(type(data)==None):
                data=new
            else:
                data=pd.concat([data,new])
    data=data.reset_index()
    data=data[["Date","Close","Stock"]]
    data["YEAR"]=data.apply(lambda row: row["Date"].year,axis=1)
    data["MONTH"]=data.apply(lambda row: row["Date"].month,axis=1)
    data["DAY_OF_MONTH"]=data.apply(lambda row: row["Date"].day,axis=1)
    data=data.drop("Date",axis=1)
    dow=_get_dow(input_dow_path)
    df=pd.concat([data,dow])
    df.to_csv(output_path,index=False)

def _match_stock(row,stock,stock_name):
    vals=stock[(stock["YEAR"]==row["YEAR"]) & (stock["MONTH"]==row["MONTH"]) & (stock["DAY_OF_MONTH"]==row["DAY_OF_MONTH"]) & (stock["Stock"]==stock_name)]["Close"].values
    if(len(vals)==0):
        return pd.NA
    else:
        return vals[0]

def _merge_bts_stock(bts_path="./data/bts_data.csv",stock_path="./data/stock_data.csv",output_path="./data/all_data.csv"):
    bts=pd.read_csv(bts_path)
    stock=pd.read_csv(stock_path)
    bts["NASDAQ"]=bts.apply(lambda row: _match_stock(row,stock,"^IXIC"),axis=1)
    bts["DOW"]=bts.apply(lambda row: _match_stock(row,stock,"DOW"),axis=1)
    bts["AIR_STOCK"]=bts.apply(lambda row: _match_stock(row,stock,CODE2STOCK[row["OP_UNIQUE_CARRIER"]]),axis=1)
    df=bts.dropna()
    df.to_csv(output_path,index=False)

def load_data(input_path="./data/all_data.csv"):
    df=pd.read_csv(input_path)
    labels=df["ARR_DEL15"].to_frame()
    features=df.drop("ARR_DEL15",axis=1)
    return features,labels

def _preproc_features_one_hot(df,max_uniques=60):
    no_1hot_cols=["AIR_STOCK","DOW","NASDAQ","DEP_TIME","CRS_DEP_TIME","DEP_DELAY","DEP_DELAY_NEW","DEP_DEL15","TAXI_OUT","WHEELS_OFF","WHEELS_ON","TAXI_IN","CRS_ARR_TIME","ARR_TIME","ARR_DELAY","ARR_DELAY_NEW","ARR_DEL15","CANCELLED","DIVERTED","CRS_ELAPSED_TIME","ACTUAL_ELAPSED_TIME","AIR_TIME","FLIGHTS","DISTANCE"]
    for col in df.columns:
        if(len(df[col].unique())<=1):
            df=df.drop(col,axis=1)
    cols=df.columns
    cols=[col for col in cols if col not in no_1hot_cols]
    for col in cols:
        if(len(df[col].unique())>=max_uniques):
            df=df.drop(col,axis=1)
        else:
            one_hot = pd.get_dummies(df[col],prefix=col)
            df = df.drop(col,axis = 1)
            df = df.join(one_hot)
    df=(df-df.mean())/df.std()
    return df.to_numpy(dtype=np.float32)

def _preproc_labels_one_hot(df):
    one_hot = pd.get_dummies(df["ARR_DEL15"],prefix="ARR_DEL15")
    df = df.drop("ARR_DEL15",axis = 1)
    df = df.join(one_hot)
    return df.to_numpy(dtype=np.float32)

def _preproc_features(df):
    no_1hot_cols=["AIR_STOCK","DOW","NASDAQ","DEP_TIME","CRS_DEP_TIME","DEP_DELAY","DEP_DELAY_NEW","DEP_DEL15","TAXI_OUT","WHEELS_OFF","WHEELS_ON","TAXI_IN","CRS_ARR_TIME","ARR_TIME","ARR_DELAY","ARR_DELAY_NEW","ARR_DEL15","CANCELLED","DIVERTED","CRS_ELAPSED_TIME","ACTUAL_ELAPSED_TIME","AIR_TIME","FLIGHTS","DISTANCE"]
    for col in df.columns:
        if(len(df[col].unique())<=1):
            df=df.drop(col,axis=1)
    cols=df.columns
    cols=[col for col in cols if col not in no_1hot_cols]
    df1=OneHotEncoder().fit_transform(df[cols])
    df2=df.drop(cols,axis=1)
    df2=(df2-df2.mean())/df2.std()
    df2=scipy.sparse.csr_matrix(df2.values)
    sp=scipy.sparse.hstack([df1,df2])
    return sp.astype(np.float32)

def _preproc_labels(df):
    return df["ARR_DEL15"].to_numpy(dtype=np.longlong)

def preproc_data(features, labels, one_hot=False):
    if(one_hot):
        return _preproc_features_one_hot(features),_preproc_labels_one_hot(labels)
    return _preproc_features(features),_preproc_labels(labels)