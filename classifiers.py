# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 18:18:04 2022

@author: vader
"""
import time
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.decomposition import *
from scipy.stats import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import *
from tqdm import tqdm
from functools import lru_cache
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import *
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC,SVR,LinearSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import *
from sklearn.manifold import *
from sklearn.metrics import *
from sklearn.base import BaseEstimator, TransformerMixin
import allel
import xgboost as xgb
from skrebate import TuRF
from ReliefF import ReliefF
from imblearn.over_sampling import RandomOverSampler,SMOTE
import pickle
from biothings_client import get_client
import warnings
warnings.filterwarnings("ignore")


def find_indices(a_list, item_to_find):
    indices = []
    for idx, value in enumerate(a_list):
        if value == item_to_find:
            indices.append(idx)
    return indices

def m_ind(inds,n_comps,c=20):

    for j in inds:
        i=len(j[0])
        if i>=(n_comps-c) and i<=(n_comps+c):
            return j




class RecDom(BaseEstimator, TransformerMixin):

        

    def fit_transform(self, X):
        W=np.zeros((X.shape[0],2*X.shape[1]),dtype='int')
        
        for i in range(X.shape[1]):
            for j in range(X.shape[0]):
                
                if X[j,i]==0:
                    W[j,i],W[j,i+1]=1,0
                elif X[j,i]==1:
                    W[j,i],W[j,i+1]=1,1
                elif X[j,i]==2:
                    W[j,i],W[j,i+1]=0,1

        return W




class Kbest_chi():
    def __init__(self,n_features):
    
    
        self.n_features=n_features
    
    def get_support(self,data,label):
        
        
        n_features=self.n_features

        with open(str('chi')+str(d_no)+str(ev_pre), "rb") as fp:   # Unpickling
            ind = pickle.load(fp)
            idx = ind[:n_features]
            return idx
        
            
        # except FileNotFoundError:
    
    
        #     rows = len(data)
        #     cols = len(data[0]) - 1
        #     T = []
        #     for j in range(0, cols):
        #         ct = [[1, 1], [1, 1], [1, 1]]
        
        #         for i in range(0, rows):
        #             if label[i] == 0:
        #                 if data[i][j] == 0:
        #                     ct[0][0] += 1
        #                 elif data[i][j] == 1:
        #                     ct[1][0] += 1
        #                 elif data[i][j] == 2:
        #                     ct[2][0] += 1
        #             elif label[i] == 1:
        #                 if data[i][j] == 0:
        #                     ct[0][1] += 1
        #                 elif data[i][j] == 1:
        #                     ct[1][1] += 1
        #                 elif data[i][j] == 2:
        #                     ct[2][1] += 1
        
        #         pval=chi2_contingency(ct)[0]
        #         T.append(pval)
        #     indices = sorted(range(len(T)), key=T.__getitem__, reverse=True)
            
        #     with open(str('chi')+str(d_no)+str(ev_pre), "wb") as fp  :
        #         pickle.dump(indices, fp)
                
        #         idx = indices[:n_features]

        # return idx

    def fit_transform(self,data,label):
        
        n_features=self.n_features
        
        try:
            with open(str('chi')+str(d_no)+str(ev_pre), "rb") as fp:   # Unpickling
                indices = pickle.load(fp)
                
                
            
            
        except FileNotFoundError:
    
    
            rows = len(data)
            cols = len(data[0]) - 1
            T = []
            for j in range(0, cols):
                ct = [[1, 1], [1, 1], [1, 1]]
        
                for i in range(0, rows):
                    if label[i] == 0:
                        if data[i][j] == 0:
                            ct[0][0] += 1
                        elif data[i][j] == 1:
                            ct[1][0] += 1
                        elif data[i][j] == 2:
                            ct[2][0] += 1
                    elif label[i] == 1:
                        if data[i][j] == 0:
                            ct[0][1] += 1
                        elif data[i][j] == 1:
                            ct[1][1] += 1
                        elif data[i][j] == 2:
                            ct[2][1] += 1
        
                pval=chi2_contingency(ct)[0]
                T.append(pval)
            indices = sorted(range(len(T)), key=T.__getitem__, reverse=True)

            
            with open(str('chi')+str(d_no)+str(ev_pre), "wb") as fp  :
                pickle.dump(indices, fp)
                
        idx = indices[:n_features]      
        
        return data[:,idx]




    def transform(self,test):
        with open(str('chi')+str(d_no)+str(ev_pre), "rb") as fp:   # Unpickling
            ind = pickle.load(fp)
            idx = ind[:self.n_features]
            return test[:,idx]
        
        
        
        
def gemisma(d1,d2):
    
    
    q=pd.DataFrame(d1)
    q1=pd.DataFrame(columns=(0,1,2))
    df=[]
    for i in tqdm(range(d1.shape[1])):
        df1=q.iloc[:,i].value_counts()*100/d1.shape[0]
        df1=pd.DataFrame(df1).transpose()
        df.append(df1)
        
    a=pd.concat(df)
    q1=pd.concat((q1,a))

    
    
    q=pd.DataFrame(d2)
    q2=pd.DataFrame(columns=(0,1,2))
    df=[]
    for i in tqdm(range(d2.shape[1])):
        df1=q.iloc[:,i].value_counts()*100/d2.shape[0]
        df1=pd.DataFrame(df1).transpose()
        df.append(df1)

    a=pd.concat(df)
    q2=pd.concat([q2,a])
    
    return q1,q2




def munneb_pre(data,y):
    
    # print(data.shape,'1')
    # a=np.where(data==-1)
    # data=np.delete(data,a[1],axis=1)
    # print(data.shape)
   

    d1=data[find_indices(y,0)]
    d2=data[find_indices(y,1)]
    
    # print(data.shape)
    
    
    
    try:
        q=pd.read_pickle(str(d_no)+str(ev_pre)+'q_mun')
    except FileNotFoundError:
        print(d1.shape)
        q1,q2=gemisma(d1,d2)
        # q=pd.concat((q1,q2),axis=1).drop([-1],axis=1)    
        q=pd.concat((q1,q2),axis=1)
        q.to_pickle(str(d_no)+str(ev_pre)+'q_mun')
    
    q.columns=('PA-PNM','PA-PPM','PA-PFM','PB-PNM','PB-PPM','PB-PFM')
    
    print(q.shape)

    q['AND']=abs(q['PA-PNM']/100*d1.shape[0]-q['PB-PNM']/100*d2.shape[0])
    q['APD']=abs(q['PA-PPM']/100*d1.shape[0]-q['PB-PPM']/100*d2.shape[0])
    q['AFD']=abs(q['PA-PFM']/100*d1.shape[0]-q['PB-PFM']/100*d2.shape[0])
    
    
    # temp=q.describe()
    # MFM=max(temp.loc['max','PA-PFM'],temp.loc['max','PB-PFM'])
    # mFM=min(temp.loc['min','PA-PFM'],temp.loc['min','PB-PFM'])
    
     
    qq=q
    
    
    # temp=qq.describe()
    # a=(qq['AND']>temp.loc['75%','AND'])
    # b=(qq['APD']>temp.loc['75%','APD'])
    # c=(qq['AFD']>temp.loc['75%','AFD'])
    
    # if d_no==1:qq=qq[a | b | c ]
    
    # print(qq.shape[1])
    
    qq['MFM']=qq[['PA-PFM','PB-PFM']].max(axis=1)
    qq['mFM']=qq[['PA-PFM','PB-PFM']].min(axis=1)
    qq['MPM']=qq[['PA-PPM','PB-PPM']].max(axis=1)
    qq['mPM']=qq[['PA-PPM','PB-PPM']].min(axis=1)
    qq['MNM']=qq[['PA-PNM','PB-PNM']].max(axis=1)
    qq['mNM']=qq[['PA-PNM','PB-PNM']].min(axis=1)
    
    # qq=q.sort_values('mFM')
    
    
    # print(qq)
    if d_no==1:l,m,n=50,180,20
    if d_no==2:l,m,n=30,100,5
    if d_no==3:l,m,n=10,2000,50
    if d_no==4:l,m,n=50,170,10
    if d_no==5:l,m,n=30,70,3
    if d_no==6:l,m,n=50,140,5
    if d_no==7:l,m,n=50,140,5
    if d_no==8:l,m,n=50,140,5




    
    
    inds=set()
    inter=0
    ind=[]
    print(str(l)+str(m)+str(n)+str(d_no)+str(ev_pre)+'inds_mun')
    try:
        with open(str(l)+str(m)+str(n)+str(d_no)+str(ev_pre)+'inds_mun', "rb") as fp:   # Unpickling
            ind = pickle.load(fp)
            return ind
    except FileNotFoundError:
            
        for j in np.arange(l,m,n):
            print(len(inds),j)
            slope=j
        
            inds=set()
            
            for i in qq.index.values:
            
                thrs=slope*qq['MFM'][i]+inter
                lthrs=(1-(thrs/100))*qq['MFM'][i]
                if qq['mFM'][i]<=lthrs:inds.add(i)
                
                thrs=slope*qq['MFM'][i]+inter
                lthrs=(1-(thrs/100))*qq['MPM'][i]
                if qq['mPM'][i]<=lthrs:inds.add(i)
            
                thrs=slope*qq['MFM'][i]+inter
                lthrs=(1-(thrs/100))*qq['MNM'][i]
                if qq['mNM'][i]<=lthrs:inds.add(i)
                
            ind.append((inds,j))
    
    
        with open(str(l)+str(m)+str(n)+str(d_no)+str(ev_pre)+'inds_mun', "wb") as fp  :
            pickle.dump(ind, fp)
    
    
        return ind





def get_data_importance(d_no,del_rows=True):
    
    
    classes_name={}
    
    classes_name[1]=['hazel','brown']
    
    classes_name[2]=['diabyes','diabno']
    
    classes_name[4]=['brown','blue']
    
    classes_name[5]=['hazel','blue']
    
    classes_name[6]=['brown','bluegrey']
    
    classes_name[7]=['brown','bluegreen']
    classes_name[8]=['brown','bgtest']

    
    if d_no!=3:
        datas=[]
        y=[]
        for i,j in enumerate(classes_name[d_no]):
            
            data=np.load(j+'.npy')
            yy=[i for k in range(data.shape[0])]
            # print(len(yy))
            y=y+yy
            datas.append(data)
            
        
        data=np.vstack(datas)
        data_y=y

    else:
        X=pd.read_csv(r'C:\Users\vader\Desktop\diplwmat\mlvsprs\mlvsprs\finalized\1\pv_5e-40\ptrain.raw', sep="\s+")
        data_y=X['PHENOTYPE'].replace([2,1],[1,0])
        data=X.iloc[:,6:]
        dataq=data.to_numpy()
        data_yq=data_y.to_numpy()
        
        X=pd.read_csv(r'C:\Users\vader\Desktop\diplwmat\mlvsprs\mlvsprs\finalized\1\pv_5e-40\ptest.raw', sep="\s+")
        data_y=X['PHENOTYPE'].replace([2,1],[1,0])
        data=X.iloc[:,6:]
        data=data.to_numpy()
        data_y=data_y.to_numpy()
        
        data=np.vstack((dataq,data))
        data_y=np.concatenate((data_yq,data_y))

        
        
        
    



    if del_rows==False:
        a=np.where(data==-1)
        data=np.delete(data,a[1],axis=1)
        data_y=np.array(data_y)
        # print(data.shape)
    else:
        # l=[]
        # for i in range(data.shape[1]):
        #     l.append(np.count_nonzero(data[:,i]==-1))

        # l=pd.Series(l)
        # print((l==0).value_counts())
        
        k=[]
        for i in range(data.shape[0]):
            k.append(np.count_nonzero(data[i,:]==-1))
        
        k=pd.Series(k)
        index=k[k > k.quantile(0.99)].index
        # print(len(index))
        data=np.delete(data,index,axis=0)
        data_y=np.delete(data_y,index,axis=0)
        a=np.where(data==-1)
        data=np.delete(data,a[1],axis=1)

        
        
        
        
        
        
    return data,data_y,a[1]

    



def get_data(d_no,del_rows=True):
    
    classes_name={}
    
    classes_name[1]=['hazel','brown']
    
    classes_name[2]=['diabyes','diabno']
    
    classes_name[4]=['brown','blue']
    
    classes_name[5]=['hazel','blue']
    
    classes_name[6]=['brown','bluegrey']
    
    classes_name[7]=['brown','bluegreen']
    classes_name[8]=['brown','bgtest']

    
    if d_no!=3:
        datas=[]
        y=[]
        for i,j in enumerate(classes_name[d_no]):
            
            data=np.load(j+'.npy')
            yy=[i for k in range(data.shape[0])]
            print(len(yy))
            y=y+yy
            datas.append(data)
            
        
        data=np.vstack(datas)
        data_y=y

    else:
        X=pd.read_csv(r'C:\Users\vader\Desktop\diplwmat\mlvsprs\mlvsprs\finalized\1\pv_5e-40\ptrain.raw', sep="\s+")
        data_y=X['PHENOTYPE'].replace([2,1],[1,0])
        data=X.iloc[:,6:]
        dataq=data.to_numpy()
        data_yq=data_y.to_numpy()
        
        X=pd.read_csv(r'C:\Users\vader\Desktop\diplwmat\mlvsprs\mlvsprs\finalized\1\pv_5e-40\ptest.raw', sep="\s+")
        data_y=X['PHENOTYPE'].replace([2,1],[1,0])
        data=X.iloc[:,6:]
        data=data.to_numpy()
        data_y=data_y.to_numpy()
        
        data=np.vstack((dataq,data))
        data_y=np.concatenate((data_yq,data_y))

        
        
        
    



    if del_rows==False:
        a=np.where(data==-1)
        data=np.delete(data,a[1],axis=1)
        data_y=np.array(data_y)
        print(data.shape)
    else:
        # l=[]
        # for i in range(data.shape[1]):
        #     l.append(np.count_nonzero(data[:,i]==-1))

        # l=pd.Series(l)
        # print((l==0).value_counts())
        
        k=[]
        for i in range(data.shape[0]):
            k.append(np.count_nonzero(data[i,:]==-1))
        
        k=pd.Series(k)
        index=k[k > k.quantile(0.99)].index
        # print(len(index))
        data=np.delete(data,index,axis=0)
        data_y=np.delete(data_y,index,axis=0)
        a=np.where(data==-1)
        data=np.delete(data,a[1],axis=1)

        
        
        
        
        
        
    return data,data_y




def pred(dpre,k,ev_pre,sample,hot,dataset):
    
    
    
    
    X,y=get_data(dataset)
    
    # from sklearn.preprocessing import StandardScaler
    # scaler = StandardScaler()
    # X=scaler.fit_transform(X)
    
    
    
    # print(X.shape)
    
    test_size=0.3
    
    if dpre=='chi':s=SelectKBest(chi2,k=k)
    if dpre=='chi':s=Kbest_chi(k)

    
    
    if dpre=='relief':s=ReliefF(n_neighbors=10, n_features_to_keep=k)
    if dpre=='pca':s=PCA(int(np.min((X.shape[0]*(1-test_size),k))),random_state=(2))
    if dpre=='rf':s = SelectFromModel(estimator=RandomForestClassifier(n_jobs=-1,random_state=3),max_features=(k)).fit(X, y)
    if dpre=='svm':
        s=SVC(random_state=3,kernel='linear',cache_size=8000).fit(X, y)
        s = SelectFromModel(estimator=s,max_features=(k)).fit(X, y)



        
        
    

    if ev_pre==False:
        
        
        
        if dpre=='mun':
            # print(X.shape)
            inds=munneb_pre(X,y)
            ind=m_ind(inds,k)
            X=X[:,list(ind[0])]
            

            if hot==True:
                
            
                enc = OneHotEncoder()
                X=enc.fit_transform(X)     
                X=X.toarray().astype('int8')   

            
        if dpre!='mun':
            
            # if hot==True:
                
            
            #     enc = OneHotEncoder()
            #     X=enc.fit_transform(X)     
            #     X=X.toarray().astype('int8')
            
            
            try:
                
                                                                  
                X=s.fit_transform(X, y)
            except UnboundLocalError:pass
        
            if hot==True:
                
                # enc = OneHotEncoder()
                enc = RecDom()
                X=enc.fit_transform(X)     
                # X=X.toarray().astype('int8')
   
    
    if sample==True:
        oversample = RandomOverSampler(sampling_strategy='minority',random_state=3)
        X,y=oversample.fit_resample(X, y)
        
        
        
    
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=test_size,random_state=2,stratify=y)
    
    # if sample==True:
    #     oversample = RandomOverSampler(sampling_strategy='minority',random_state=3)
    #     X_train,y_train=oversample.fit_resample(X_train, y_train)
       
    
    if ev_pre==True:
        
        if dpre=='mun':
            # print(X_train.shape)
            inds=munneb_pre(X_train,y_train)
            ind=m_ind(inds,k)
            X_train=X_train[:,list(ind[0])]
            X_test=X_test[:,list(ind[0])]
            
            if hot==True:
                
            
                enc = OneHotEncoder(handle_unknown='ignore')
                X_train=enc.fit_transform(X_train).toarray().astype('int8')  
                X_test=enc.transform(X_test).toarray().astype('int8')  

        
        if dpre=='pca':
            s=PCA(int(np.min((X_train.shape[0],k))),random_state=(2))
            

        try:
            
            if hot==True:
                
            
                enc = OneHotEncoder(handle_unknown='ignore')
                X_train=enc.fit_transform(X_train).toarray().astype('int8')  
                X_test=enc.transform(X_test).toarray().astype('int8')  

                
            X_train=s.fit_transform(X_train,y_train)
            X_test=s.transform(X_test)
        except NameError: pass
       
        
        
    
    
    print(X_train.shape)
    






    print(dpre)
    
    start = time.time()
    
    clf1=SVC(random_state=3,cache_size=8000)
    clf1.fit(X_train,y_train)
    stop = time.time()
    
    clf1st,clf1stt=clf1.score(X_train,y_train),clf1.score(X_test,y_test)
    print(clf1st,clf1stt,f"Training time: {stop - start}s")
    

    
    
    # clf=RidgeClassifier(random_state=4)
    # clf.fit(X_train,y_train)
    # print(clf.score(X_train,y_train),clf.score(X_test,y_test))
    
    start = time.time()
    
    
    clf2=RandomForestClassifier(random_state=4,n_jobs=-1)
    clf2.fit(X_train,y_train)
    stop = time.time()


    clf2st,clf2stt=clf2.score(X_train,y_train),clf2.score(X_test,y_test)
    print(clf2st,clf2stt,f"Training time: {stop - start}s")
    
    
    apotelesmata.loc[len(apotelesmata.index)]=[dpre,X_train.shape[1],clf1st,clf1stt,
                         clf2st,clf2stt,
                         hot,ev_pre,dataset]


    # return print(clf.score(X_train,y_train),clf.score(X_test,y_test))



def save_reset(name,apotelesmata):
    apotelesmata.to_csv('results/'+name+'.csv')
    apotelesmata= pd.DataFrame(columns=['Method','N_features','Train_acc1','Test_acc1','Train_acc2','Test_acc2','One-hot','Ev-pre','Dataset'])
    return apotelesmata


def see_inds(d):
    d_no=d
    if d_no==1:l,m,n=50,180,20
    if d_no==2:l,m,n=30,100,5
    if d_no==3:l,m,n=10,2000,50
    if d_no==4:l,m,n=50,170,10
    if d_no==5:l,m,n=30,70,3
    if d_no==6:l,m,n=50,140,5
    if d_no==7:l,m,n=50,140,5
    if d_no==8:l,m,n=50,140,5
    
    with open(str(l)+str(m)+str(n)+str(d_no)+str(ev_pre)+'inds_mun', "rb") as fp:   # Unpickling
        ind = pickle.load(fp)
    for j in ind:
         i=len(j[0])
         print(i)
    
        





# global apotelesmata 
apotelesmata= pd.DataFrame(columns=['Method','N_features','Train_acc1','Test_acc1','Train_acc2','Test_acc2','One-hot','Ev-pre','Dataset'])




"""
pre

"""


pres=('chi','pca','relief','mun','no')
ev_pre=True




# d_no=1
# for j in pres:
    
#     for i in (67,126,309):
#         if i<200 and j=='no':continue
#         pred(j,i,ev_pre=ev_pre,sample=True,hot=False,dataset=d_no)  



# apotelesmata=save_reset('1pre',apotelesmata)



# d_no=2
# for j in pres:
#     for i in (18,53,130,244):
    
#         pred(j,i,ev_pre=ev_pre,sample=True,hot=False,dataset=d_no)  

# apotelesmata=save_reset('2pre',apotelesmata)




# d_no=4
# for j in pres:
#     for i in (77,134,273,424):
#         if i<200 and j=='no':continue
#         pred(j,i,ev_pre=ev_pre,sample=True,hot=False,dataset=d_no)  

# apotelesmata=save_reset('4pre',apotelesmata)


# d_no=5
# for j in pres:
#     for i in (102,183,247):
#         if i<200 and j=='no':continue

#         pred(j,i,ev_pre=ev_pre,sample=True,hot=False,dataset=d_no)  


# apotelesmata=save_reset('5pre',apotelesmata)

# d_no=6
# for j in pres:
#     for i in (172,236,459):
    
#         pred(j,i,ev_pre=ev_pre,sample=True,hot=False,dataset=d_no)  

# apotelesmata=save_reset('6pre',apotelesmata)




# d_no=7
# for j in pres:
#     for i in (140,251,457):
    
#         pred(j,i,ev_pre=ev_pre,sample=True,hot=False,dataset=d_no)  

# apotelesmata=save_reset('7pre',apotelesmata)



# d_no=8
# for j in pres:
#     for i in (98,151,246,532):
    
#         pred(j,i,ev_pre=ev_pre,sample=True,hot=False,dataset=d_no)  

# apotelesmata=save_reset('8pre',apotelesmata)


"""
encoding
"""

# ev_pre=False
# j='no'
# i=1

# for d_no in (1,2,4,5,6,7):
#     for hot in (True,False):
#         pred(j,i,ev_pre=ev_pre,sample=True,hot=hot,dataset=d_no)  
  
    
# apotelesmata=save_reset('encoding',apotelesmata)








""" 


classification

"""



# for d_no in (1,2,4,5,6,7):
#     X,y=get_data(d_no)
#     oversample = RandomOverSampler(sampling_strategy='minority',random_state=3)
#     X,y=oversample.fit_resample(X, y)
#     X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3,random_state=2,stratify=y)
    
    
#     classifiers = [
#         SVC(random_state=3,cache_size=8000),
    
#         RandomForestClassifier(n_jobs=-1,random_state=3),
    
#     ]
    
#     for clf in classifiers:
#         clf.fit(X_train,y_train)
#         print(clf,clf.score(X_test,y_test))










"""
Feature importance

"""



# def f(k,d):
#     print(d)
#     X,y,a=get_data_importance(d)
#     snps=np.load('snps_001-10.npy',allow_pickle=True)
#     snps=np.delete(snps,a)
#     snps=pd.DataFrame(snps)
    
    

    # inds=munneb_pre(X,y)
    # ind=m_ind(inds,k,c=1000)
    # i=list(ind[0])
    # important_mun=snps.iloc[i]
    # final=list(set(important_mun[0]))

    
    
    
    
    # s = SelectFromModel(estimator=RandomForestClassifier(n_jobs=-1,random_state=3),max_features=(k)).fit(X, y)
    # i=s.get_support()
    # important_rf=snps.iloc[i]
    # final=list(set(important_rf[0]))
    
    
    # s=SVC(random_state=3,kernel='linear',cache_size=8000).fit(X, y)
    # s = SelectFromModel(estimator=s,max_features=(k)).fit(X, y)
    # i=s.get_support()
    # important_svm=snps.iloc[i]
    # final=list(set(important_svm[0]))
    
    
    # final=list(set(important_rf[0])|set(important_svm[0]))

    
    
    
    
    
    
    
    # s=SelectKBest(f_regression,k=k).fit(X,y)
    # s=Kbest_chi(k)
    # i=s.get_support(X,y)
    # important_chi=snps.iloc[i]
    # final=list(set(important_chi[0]))
    
    
    
    # i=ReliefF(n_neighbors=10, n_features_to_keep=k).get_features(X,y)
    
    # important_relief=[]
    # for j in i:        
    #     important_relief.append(snps[0][j])
    # final=set(important_relief)
   
    
    



    
    
    
    
    
    
    
    
#     final=[i.partition('_')[0] for i in final]
#     final=[i.partition(';')[0] for i in final]
    
    
#     try:
#         oll_genes=set(pd.read_pickle('oll'))
#     except FileNotFoundError:
#         oll=set(snps[0])
#         oll=[i.partition('_')[0] for i in oll]
#         oll=[i.partition(';')[0] for i in oll]
    
        
#         vm = {x : None for x in oll}
#         vc = get_client('variant')
#         rs = vc.querymany(oll, scopes='dbsnp.rsid', fields='all', verbose=True)
#         for r in rs:
#             q = r['query']
#             if 'dbsnp' in r:
#                 z = r['dbsnp']
#                 if 'gene' in z:
#                     g = z['gene']
#                     if 'symbol' in g and not vm[q]:
#                         vm[q] = g['symbol']
    
    
        
#         vm=list(vm.values())
#         vm=[i for i in vm if isinstance(i, str) ]
#         vm=set(vm)
#         oll_genes=vm
#         pd.Series(list(oll_genes)).to_pickle('oll')





    
    
#     vm = {x : None for x in final}
#     vc = get_client('variant')
#     rs = vc.querymany(final, scopes='dbsnp.rsid', fields='all', verbose=False)
#     for r in rs:
#         q = r['query']
#         if 'dbsnp' in r:
#             z = r['dbsnp']
#             if 'gene' in z:
#                 g = z['gene']
#                 if 'symbol' in g and not vm[q]:
#                     vm[q] = g['symbol']
                    

    
    
#     vm=list(vm.values())
#     vm=[i for i in vm if isinstance(i, str) ]
#     vm=set(vm)
#     # print(np.unique(vm))
#     genes=vm
    
    
   
#     if d!=2:
#         obs_genes={'ASIP','IRF4','SLC24A4','SLC24A5','SLC24A2','TPCN2','TYR','TYRP1','OCA2','HERC2'}
#     else:
        
#         obs_genes={'TCF7L2','ABCC8','CAPN10','GLUT2','GCGR'}
    
    
#     print('koina genes tou dataset',obs_genes & oll_genes)

#     print()
#     print ('koina',genes & obs_genes,len((vm)))






# ev_pre=True
# k=1000
# d_no=1
# f(k,d_no)
# # d_no=2
# # f(k,d_no)
# d_no=4
# f(k,d_no)
# d_no=5
# f(k,d_no)
# d_no=6
# f(k,d_no)
# d_no=7
# f(k,d_no)





#feature selection support
d_no=1
for i in (67,126,309):
    pred('svm',i,ev_pre=False,sample=True,hot=False,dataset=d_no)  
    pred('rf',i,ev_pre=False,sample=True,hot=False,dataset=d_no)  
    
d_no=4
for i in (77,134,273,424):
    pred('svm',i,ev_pre=False,sample=True,hot=False,dataset=d_no)  
    pred('rf',i,ev_pre=False,sample=True,hot=False,dataset=d_no)  
    
d_no=5
for i in (102,183,247):
    pred('svm',i,ev_pre=False,sample=True,hot=False,dataset=d_no)  
    pred('rf',i,ev_pre=False,sample=True,hot=False,dataset=d_no)  

d_no=6
for i in (172,236,459):
    pred('svm',i,ev_pre=False,sample=True,hot=False,dataset=d_no)  
    pred('rf',i,ev_pre=False,sample=True,hot=False,dataset=d_no)  
    
d_no=7
for i in (140,251,457):
    pred('svm',i,ev_pre=False,sample=True,hot=False,dataset=d_no)  
    pred('rf',i,ev_pre=False,sample=True,hot=False,dataset=d_no)  





