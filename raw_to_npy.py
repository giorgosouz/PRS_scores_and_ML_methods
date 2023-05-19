# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# import required modules
import pandas as pd
import numpy as np
import time
from dask import dataframe as df1
from tqdm import tqdm




df=pd.read_csv('001_-10.raw', sep="\s+",nrows=1)
cols=df.columns



df=pd.read_csv('001_-10.raw', sep="\s+",usecols=['FID'])
tags=df.values


















chunkie=pd.read_csv('001_-10.raw', sep="\s+",usecols=cols[6:],chunksize=(100))


l=[]

for k,i in enumerate(chunkie):
    print(k*100,df.shape[0])
    i.fillna(-1,inplace=True)
    l.append(i)


c=pd.concat(l)




snps=df.iloc[0,6:].index.values
np.save('snps_001_-10.npy',snps)

c=c.to_numpy(dtype=('int8'))
np.save('001_-10.npy',c)
np.save('tags_001_-10.npy',tags)
np.save('snps_001-10.npy',cols[6:])


data=np.load('001_-10.npy')
tags=np.load('tags_001_-10.npy')

classes_name=['blue','brown','green','bluegreen','bluegrey','hazel','bgtest','diabyes','diabno']


datas=[]
for i in classes_name:
    
    tag=np.loadtxt(i+'.txt',dtype=int)
    
    
    qq=np.where(tag==tags)
    
    fin=data[list(qq[0]),:]
    
    
    # # l=[]
    # # for i in range(fin.shape[1]):
    # #     l.append(np.count_nonzero(fin[:,i]==-1))

    # # l=pd.Series(l)
    # # print((l==0).value_counts())
    
    # k=[]
    # for j in range(fin.shape[0]):
    #     k.append(np.count_nonzero(fin[j,:]==-1))
    
    # k=pd.Series(k)
    # index=k[k > k.quantile(0.97)].index
    # # print(len(index))
    # fin=np.delete(fin,index,axis=0)
    # # fin_y=np.delete(fin_y,index,axis=0)
    # a=np.where(fin==-1)
    # fin=np.delete(fin,a[1],axis=1)

    
    
    np.save(i+'.npy',fin)
    datas.append(fin)




""" 

l=[]
for i in range(data.shape[1]):
    l.append(np.count_nonzero(data[:,i]==-1))

l=pd.Series(l)
(l==0).value_counts()
    
    
"""