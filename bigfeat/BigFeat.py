#!/usr/bin/env python
# coding: utf-8

# In[3]:


from bigfeat_base import BigFeat
from local_utils import *
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, make_scorer
import os
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import f1_score
from autofeat import AutoFeatClassifier
import matplotlib.pyplot as plt
import os


# In[3]:





# In[5]:


baseDir=r'/home/ubuntu/BigFeat-master/bigfeat/AutoMl_datasets/New folder'
datasets=os.listdir(baseDir)
ml_models={"Logistic_Regression":LogisticRegression(),
           "AdaBoostClassifier":AdaBoostClassifier(n_estimators=100, random_state=0),
           "DecisionTreeClassifier":DecisionTreeClassifier(random_state=0, max_depth=5),
           "ExtraTreesClassifier":ExtraTreesClassifier(n_estimators=100, random_state=0),
           "KNNClassifier":KNeighborsClassifier(n_neighbors=3),
           "MLP":MLPClassifier(random_state=1, max_iter=300),
           "RandomForestClassifier":RandomForestClassifier(max_depth=2, random_state=0),
           "SVM":svm.SVC(kernel='linear',probability=True),
           "GradientBoostingClassifier":GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)
          }
    


# In[27]:



iterables = [[i.split('.')[0] for i in datasets], ["BigFeat","AutoFeat"],["Original", "Generated"],['F1_score','AUC']]

index = pd.MultiIndex.from_product(iterables, names=["DataSet","Framework","Type","Evaluation"])

dataframe= pd.DataFrame(columns=[i for i in ml_models], index=index)
    
dataframe


# In[ ]:





# In[6]:


dataframe.loc[('phoneme','BigFeat','Original')]


# In[28]:


def getData(directory):
    data=pd.read_csv(directory)
    data=data.dropna() #  For the simplicity, drop the null values from dataset
    y=data.Label
    X=data.drop('Label',axis=1)
    return X,y


# In[29]:


def getBarPlot(dataframe,evaluation_metric):

    N=len(dataframe.columns)

    x1=dataframe.loc[('phoneme','BigFeat','Original',evaluation_metric)].tolist()
    x2=dataframe.loc[('phoneme','BigFeat','Generated',evaluation_metric)].tolist()
    x3=dataframe.loc[('phoneme','AutoFeat','Generated',evaluation_metric)].tolist()

    for j in range(N):
    
        fig = plt.figure()
        ax = fig.add_axes([0,0,1,1])
        features=['Original','BigFeat','AutoFeat']
        values=[x1[j],x2[j],x3[j]]
    
        plt.ylabel(evaluation_metric + ' score')
        plt.xlabel('Features')
        plt.title(dataframe.columns[j])
    
        bars=plt.bar(features,values,color = 'b', width = 0.3)
        
        
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x(), round(yval,5)+.010, round(yval,5))
            
        plt.show()
    


# In[ ]:





# In[30]:


def mlPipeline (X,y,ml_models,data_name,Framework):
    
    # This data's for significance analysis
    data_Container={}
    
    
    if Framework=='BigFeat':
        data_BigFeat={}
        data_Original={}
        
        #Feature generation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
            
        feature_generation=BigFeat()
        X_train_tr=feature_generation.fit(X_train,y_train)
    
    
    
        for i in ml_models:
            
            #Generated features
            
            model=ml_models[i]
            model.fit(X_train_tr,y_train)
          
            
            if len(y.unique())>2:   # For multiclass classification
                y_pred=model.predict_proba(feature_generation.transform(X_test))
                data_BigFeat[i]=y_pred
                try: 
                    score_auc=roc_auc_score(y_test,y_pred,multi_class="ovr")
                    score_f1=f1_score(y_test, y_pred, average='weighted')
                except ValueError:
                    pass
               
                dataframe.loc[(data_name,Framework,"Generated","AUC")][i]=score_auc
                dataframe.loc[(data_name,Framework,"Generated","F1_score")][i]=score_f1
                #dataframe.loc[(data_name,"Generated")]['Framework']=Framework
            else :
                y_pred=model.predict(feature_generation.transform(X_test))
                data_BigFeat[i]=y_pred
                try:
                    score_auc=roc_auc_score(y_test,y_pred)
                    score_f1=f1_score(y_test, y_pred)
                except ValueError:
                    pass
                dataframe.loc[(data_name,Framework,"Generated",'AUC')][i]=score_auc
                dataframe.loc[(data_name,Framework,"Generated",'F1_score')][i]=score_f1
                #dataframe.loc[(data_name,"Generated")]['Framework']=Framework
                data_Container[Framework]=data_BigFeat
        
            
    
            #originial Feature
        
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
            model=ml_models[i]
            model.fit(X_train,y_train)
            
            
            
            if len(y.unique())>2:   # For multiclass classification
                
                y_pred=model.predict_proba(X_test)
                data_Original[i]=y_test
                try:        # if tartget variable extremly unbias 
                    score_auc=roc_auc_score(y_test,y_pred,multi_class="ovr")
                    score_f1=f1_score(y_test, y_pred, average='weighted')
                except ValueError:
                    pass
                
                dataframe.loc[(data_name,Framework,"Original","AUC")][i]=score_auc
                dataframe.loc[(data_name,Framework,"Original","F1_score")][i]=score_f1
                #dataframe.loc[(data_name,Framework,"Original")]['Framework']=Framework
            else :
                y_pred=model.predict(X_test)
                data_Original[i]=y_test
                try:
                    score_auc=roc_auc_score(y_test,y_pred)
                    score_f1=f1_score(y_test, y_pred)
                except ValueError:
                    pass
                dataframe.loc[(data_name,Framework,"Original","AUC")][i]=score_auc
                dataframe.loc[(data_name,Framework,"Original","F1_score")][i]=score_f1
               # dataframe.loc[(data_name,"Original")]['Framework']=Framework
            
            
            data_Container['Original']=data_Original
    
    if Framework=='AutoFeat':
        data_AutoFeat={}
        
        #Feature generation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
            
        feature_generation=AutoFeatClassifier(verbose=0, feateng_steps=1,featsel_runs=1)
        X_train_tr=feature_generation.fit_transform(X_train,y_train)
    
        
        
        for i in ml_models:
        
            #Generated features
            
            model=ml_models[i]
            model.fit(X_train_tr,y_train)
          
            
            if len(y.unique())>2:   # For multiclass classification
                y_pred=model.predict_proba(feature_generation.transform(X_test))
                data_AutoFeat[i]=y_pred
                try: 
                    score_auc=roc_auc_score(y_test,y_pred,multi_class="ovr")
                    score_f1=f1_score(y_test, y_pred, average='weighted')
                except ValueError:
                    pass
                
                dataframe.loc[(data_name,Framework,"Generated","AUC")][i]=score_auc
                dataframe.loc[(data_name,Framework,"Generated","F1_score")][i]=score_f1
                #dataframe.loc[(data_name,"Generated")]['Framework']=Framework
            else :
                y_pred=model.predict(feature_generation.transform(X_test))
                data_AutoFeat[i]=y_pred
                try:
                    score_auc=roc_auc_score(y_test,y_pred)
                    score_f1=f1_score(y_test, y_pred)
                except ValueError:
                    pass
                dataframe.loc[(data_name,Framework,"Generated","AUC")][i]=score_auc
                dataframe.loc[(data_name,Framework,"Generated","F1_score")][i]=score_f1
                #dataframe.loc[(data_name,"Generated")]['Framework']=Framework
                data_Container[Framework]=data_AutoFeat
        
            #originial Feature
        
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
            model=ml_models[i]
            model.fit(X_train,y_train)
            
            
            if len(y.unique())>2:   # For multiclass classification
                
                y_pred=model.predict_proba(X_test)
                try:        # if tartget variable extremly unbias 
                    score_auc=roc_auc_score(y_test,y_pred,multi_class="ovr")
                    score_f1=f1_score(y_test, y_pred, average='weighted')
                    
                except ValueError:
                    pass
                dataframe.loc[(data_name,Framework,"Original","AUC")][i]=score_auc
                dataframe.loc[(data_name,Framework,"Original","F1_score")][i]=score_f1
                #dataframe.loc[(data_name,"Original")]['Framework']=Framework
            else :
                y_pred=model.predict(X_test)
                try:
                    score_auc=roc_auc_score(y_test,y_pred)
                    score_f1=f1_score(y_test, y_pred)
                except ValueError:
                    pass
                dataframe.loc[(data_name,Framework,"Original","AUC")][i]=score_auc
                dataframe.loc[(data_name,Framework,"Original","F1_score")][i]=score_f1
                #dataframe.loc[(data_name,"Original")]['Framework']=Framework
                
    #createExcel(data_Container,data_name,ml_models)            


# In[10]:


def startPipeline(directory,datasets,ml_models,framework):
    
    for dataset in datasets:
        start=time.time()
        X,y=getData(f'{directory}/{dataset}')
        
        mlPipeline(X,y,ml_models,dataset.split('.')[0],framework)
        stop=time.time()
        print("{} seconds taken for the dataset {}".format(stop-start,dataset))
    return dataframe


# In[36]:


def createExcel(data,data_name,ml_models):
    
    parent_dir=r'/home/ubuntu/BigFeat-master/bigfeat/Excels'
    path = os.path.join(parent_dir, data_name)
    
    if not os.path.exists(path):
        os.makedirs(path)

    BigFeat=data.get('BigFeat')
    Original=data.get('Original')
    AutoFeat=data.get('AutoFeat')
    
    for model in ml_models:
        dataFrame=pd.DataFrame(columns=['Original','BigFeat','AutoFeat'])
        if BigFeat !=None:
            if not os.path.exists(f'{path}/{model}.xlsx'):
                dataFrame['Original']=Original[model]
                dataFrame['BigFeat']=BigFeat[model]
                dataFrame.to_excel(f'{path}/{model}.xlsx')
            else:
                dataFrame=pd.read_excel(f'{path}/{model}.xlsx')
                dataFrame['Original']=Original[model]
                dataFrame['BigFeat']=BigFeat[model]
                dataFrame.to_excel(f'{path}/{model}.xlsx')
        if AutoFeat != None:
            
            if not os.path.exists(f'{path}/{model}.xlsx'):
                dataFrame['AutoFeat']=AutoFeat[model]
                dataFrame.to_excel(f'{path}/{model}.xlsx')
            else:
                dataFrame=pd.read_excel(f'{path}/{model}.xlsx')
                dataFrame['AutoFeat']=AutoFeat[model]
                dataFrame.to_excel(f'{path}/{model}.xlsx')


# In[40]:


startPipeline(baseDir,datasets,ml_models,'BigFeat')


# In[37]:


startPipeline(baseDir,datasets,ml_models,'AutoFeat')


# In[29]:


dataframe


# In[ ]:




