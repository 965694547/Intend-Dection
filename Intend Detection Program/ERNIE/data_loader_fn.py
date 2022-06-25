import pandas as pd
from paddlenlp.datasets import MapDataset

def loader(data,labelnum,taskname):
    df = pd.read_csv('./data/'+data+'/'+taskname+'.csv')
    ds= []
    for i in range(len(df)):
        dictionary={'text':df.iloc[i]['text'],'label':df.iloc[i]['labelIndex'],'qid':i}
        ds.append(dictionary)
    ds=MapDataset(ds)
    ds.label_list=list(range(labelnum))
    return ds

def data_loader_process(data):
    df = pd.read_csv('./data/'+data+'/train.csv')
    labelName = df.intent.unique()# 全部label列表

    return loader(data,len(labelName),'train'),loader(data,len(labelName),'train'),loader(data,len(labelName),'dev'),loader(data,len(labelName),'test')