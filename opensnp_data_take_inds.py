# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 16:23:01 2022

@author: vader
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import os
from zipfile import ZipFile
import re






def search_file(onomasia):


   return onomasia.split('.')[0]






data=pd.read_csv("C:/Users/vader/Desktop/phenotypes_202209231154.csv",
                 sep=';')




# re.findall(r'\b\d+\b',a)


# #see all categories ranked by max data

# l,m,n=[],[],[]
# for col in data.columns:
#     try:
#         tmp=np.sort(np.unique(data[col],return_counts=True)[1])
#         l.append(col)
#         m.append(list(tmp[-5:]))
#         n.append(np.sum(tmp[-5:-1]))
        
#     except TypeError:continue

# df=pd.DataFrame((l,m,n)).transpose().sort_values(by=2,ascending=False)
# df.head(10)


#check the classes
eye1=data['Eye Color']
ueye1=np.unique(eye1,return_counts=True)
temp=np.column_stack((ueye1))

eye2=data['Eye color']
ueye2=np.unique(eye2,return_counts=True)
temp=np.column_stack((ueye2))


df=pd.DataFrame([eye1,eye2]).transpose()



diab1=data['Type II Diabetes']
udiab1=np.unique(diab1,return_counts=True)
temp=np.column_stack((udiab1))


diab2=data['Type 2 Diabetes +rs13266634']
udiab2=np.unique(diab2,return_counts=True)
temp=np.column_stack((udiab2))


# eyes=pd.concat([eye1,eye2],axis=1)
# diabetes=pd.concat([diab1,diab2],axis=1)


ast=data['Astigmatism']
uast=np.unique(ast,return_counts=True)
temp=np.column_stack((uast))










#how to extract





with ZipFile('E:/data/dna/opensnp/opensnp_datadump.current.zip', 'r') as zipObj:
    # Get list of files names in zip
    listOfiles = zipObj.namelist()
    # Iterate over the list of file names in given list & print them














eye1=data[['Eye Color','genotype_filename']]
eye2=data[['Eye color','genotype_filename']]
eye1=eye1.rename(columns={'Eye Color':'Eye color'},inplace=True)






a=(eye2['Eye color']=='Blue')
b=(eye2['Eye color']=='Blue-green')
c=(eye2['Eye color']=='Green')
d=(eye2['Eye color']=='blue')
e=(eye2['Eye color']=='green')
f=(eye2['Eye color']=='Brown')
g=(eye2['Eye color']=='Blue-green ')
h=(eye2['Eye color']=='blue-green ')
i=(eye2['Eye color']=='blue-green')
j=(eye2['Eye color']=='brown')
k=(eye2['Eye color']=='Blue-grey')
l=(eye2['Eye color']=='Hazel')
m=(eye2['Eye color']=='Hazel (brown/green)')
o=(eye2['Eye color']=='hazel')
p=(eye2['Eye color']=='Green ')







bluegreen=(b|g|h|i).index[b|g|h|i]
blue=(a|d).index[a|d]
green=(c|e|p).index[c|e|p]
brown=(f|g).index[f|g]
bluegrey=(k).index[k]
hazel=(l|m|o).index[l|m|o]
bgtest=(a|d|c|e|p|b|g|h|i).index[a|d|c|e|p|b|g|h|i]





diab=data[['Type II Diabetes','genotype_filename']]



a=(diab['Type II Diabetes']=='Diabetes Mellitus')
b=(diab['Type II Diabetes']=='diabetes mellitus')
c=(diab['Type II Diabetes']=='Gestational Diabetic and now Type II')
d=(diab['Type II Diabetes']=='Gestational diabetic and now type ii')
e=(diab['Type II Diabetes']=='Yes')




diabyes=(a|b|c|d|e).index[(a|b|c|d|e)]


a=(diab['Type II Diabetes']=='No')
b=(diab['Type II Diabetes']=='no')



diabno=(a|b).index[a|b]








classes=[blue,brown,green,bluegreen,bluegrey,hazel,bgtest,diabyes,diabno]

fname=data['genotype_filename']

elements=[]
for i in classes:
    element=[]
    for j in i:
        k=fname[j]
        res=search_file(k)
        element.append(res)
        # print(k,'------------------',res)

    elements.append(element)



classes_name=['blue','brown','green','bluegreen','bluegrey','hazel','bgtest','diabyes','diabno']

# with ZipFile('E:/New folder (3)/opensnp_datadump.current.zip', 'r') as zipObj:
d=os.getcwd()







for i,j in enumerate(elements):
    
    with open(d+os.sep+str(classes_name[i])+'.txt','w') as fp:
       
        for item in j:
            # write each item on a new line
            fp.write("%s\n" % item)






