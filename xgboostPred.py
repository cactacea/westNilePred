
import numpy as np
import sklearn.linear_model as linear
import sklearn.ensemble as ensemble
from sklearn import cross_validation
import pandas as pd
import sys
from collections import deque
import random
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import scipy.ndimage.filters as flt
from collections import defaultdict
#from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVR
import pickle as pk
import myutils
import itertools
from sklearn.metrics import roc_curve, auc
import pylab as P
import math
import sys
sys.path.append('C:/Users/Pasad/Downloads/xgboost-master/wrapper')
import xgboost as xgb
algos = ['bayesianRidge,adaBoost','decisionTree','gradBoost','extraTree','linear']


#All static data structures here
clf  = defaultdict.fromkeys(algos)
master_obj = master()


#All parameters go here.

feat_list = 'all'
learner_id=0 
#run_mode = 'CV'
run_mode = 'CV'
if run_mode == 'CV':
    cv_factor = 0.7
    num_cv_folds =5
else:
    cv_factor = 1
    num_cv_folds =1

#Feature engineering
#basepath = 'C:/Users/kpasad/data/ML/projects/westNilePred/'
basepath = 'D:/ml/projects/westNile_pred/'
df_train = pd.read_csv(basepath+'train.csv')
df_weather = pd.read_csv(basepath+'weather.csv')
df_test = pd.read_csv(basepath+'test.csv')
if run_mode == 'submit':
    submissionFile = open(basepath+'submissionFile.csv','w')

#Train file feature engg:
year = pd.DatetimeIndex(pd.DatetimeIndex(df_train['Date'])).year
months = pd.DatetimeIndex(pd.DatetimeIndex(df_train['Date'])).month #1st convertion to datetime64 from object. Next to date format. 
days =pd.DatetimeIndex(pd.DatetimeIndex(df_train['Date'])).day
df_train['year'] = pd.Series(year, index=df_train.index) #Add column to data frame
df_train['month'] = pd.Series(months, index=df_train.index) #Add column to data frame
df_train['day'] = pd.Series(days, index=df_train.index) #Add column to data frame



#weather_feats =['Station', 'Date','Tmax', 'Tmin', 'Tavg', 'Depart', 'DewPoint', 'WetBulb', 'Heat', 'Cool','Sunrise', 'Sunset', 'CodeSum', 'Depth', 'Water1', 'SnowFall', 'PrecipTotal', 'StnPressure', 'SeaLevel', 'ResultSpeed', 'ResultDir', 'AvgSpeed']
weather_feats_to_remove=['Station','Tmax', 'Tmin', 'Sunrise', 'Sunset', 'CodeSum', 'Depth', 'Water1', 'SnowFall', 'PrecipTotal', 'StnPressure', 'SeaLevel', 'ResultSpeed', 'ResultDir', 'AvgSpeed','WetBulb']
weather=df_weather[df_weather['Station']==1].drop(weather_feats_to_remove,axis=1)
listOfWeatherFeatsToFloatify = weather.columns.tolist()
listOfWeatherFeatsToFloatify.remove('Date')
weather[listOfWeatherFeatsToFloatify] = weather[listOfWeatherFeatsToFloatify].astype(float)
df_master_train = df_train.merge(weather)

#Test file feature engg:
year = pd.DatetimeIndex(pd.DatetimeIndex(df_test['Date'])).year #1st convertion to datetime64 from object. Next to date format. 
months = pd.DatetimeIndex(pd.DatetimeIndex(df_test['Date'])).month #1st convertion to datetime64 from object. Next to date format. 
days =pd.DatetimeIndex(pd.DatetimeIndex(df_test['Date'])).day
df_test['year'] = pd.Series(year, index=df_test.index) #Add column to data frame
df_test['month'] = pd.Series(months, index=df_test.index) #Add column to data frame
df_test['day'] = pd.Series(days, index=df_test.index) #Add column to data frame

df_master_test = df_test.merge(weather)

#train_feats = ['year','month','day','Latitude', 'Longitude']
train_feats = ['year','month','day','Latitude', 'Longitude', 'Tavg', 'Depart', 'DewPoint', 'Heat', 'Cool']
x_train =df_master_train[train_feats]
y_train=df_master_train['WnvPresent'] #Extract the targets
x_test =df_master_test[train_feats]




#Bootstrap parameters
n_train = df_train.shape[0]
train_sample_idx = range(0,n_train)
num_cv_train_samples = int(df_master_train.shape[0]*cv_factor)
train_sample_idx = deque(range(0,n_train))

param = {}
# use logistic regression loss, use raw prediction before logistic transformation
# since we only need the rank
param['objective'] = 'binary:logitraw'
# scale weight of positive examples
param['scale_pos_weight'] = 10
param['eta'] = 0.5
param['max_depth'] = 2
param['eval_metric'] = 'auc'
param['silent'] = 1
param['min_child_weight'] = 50
param['subsample'] = 1
param['colsample_bytree'] = 0.5
param['nthread'] = 1
param['verbose'] = 2

num_round = 50


for year_num in range(2007,2014,2):
              
    x_train_cv = x_train[x_train['year']==year_num]                                
    y_train_cv = y_train[x_train['year']==year_num]

    dtrain = xgb.DMatrix(x_train_cv.values, label=np.squeeze(y_train_cv.values),missing=999)
    bst = xgb.Booster() #init model 
    bst = xgb.train(param, dtrain, num_round)
    bst.save_model('model_'+str(year_num))

if (run_mode == 'CV') :
    mean_roc=0
    predictor_cnt=0    
    for year_num in range(2007,2014,2):
        #All other years are for L2 data.          
        x_train_cv = x_train[x_train['year']!=year_num]                                
        y_train_cv = y_train[x_train['year']!=year_num]
        pred=np.zeros((x_train_cv.shape[0],3))    
    
        dtrain_l1 = xgb.DMatrix(x_train_cv.values,missing=999)
 
        listOfModels=range(2007,2014,2)
        listOfModels.remove(year_num)
        modelCnt=0
        for l2_model in listOfModels:
            bst = xgb.Booster() #init model
            bst.load_model('model_'+str(l2_model))
            pred[:,modelCnt]= 1/(1+exp(bst.predict(dtrain_l1)))
            modelCnt=modelCnt+1
  
        param['scale_pos_weight'] = 10
        param['eta'] = 0.5
        param['max_depth'] = 2
        param['eval_metric'] = 'auc'
        param['silent'] = 1
        param['min_child_weight'] = 50
        param['subsample'] = 1
        param['colsample_bytree'] = 1
        param['nthread'] = 1
     
        dtrain = xgb.DMatrix(pred, label=np.squeeze(y_train_cv.values),missing=999)
        bst = xgb.Booster() #init model
        bst = xgb.train(param, dtrain, num_round)
        bst.save_model('l2model')        
        x_test_cv =x_train[x_train['year']==year_num]
        y_test_cv =y_train[x_train['year']==year_num] 
        dtest = xgb.DMatrix(x_test_cv.values,missing=999)
        #L1 predictions
        modelCnt=0 
        pred_test=np.zeros((x_test_cv.shape[0],3))    

        for l2_model in listOfModels:
            bst = xgb.Booster() #init model
            bst.load_model('model_'+str(l2_model))
            pred_test[:,modelCnt]= 1/(1+exp(bst.predict(dtest)))
            modelCnt=modelCnt+1
        dtest = xgb.DMatrix(pred_test,missing=999)

        bst = xgb.Booster() #init model
        bst.load_model('l2model')
        cv_pred = 1/(1+exp(bst.predict(dtest)))
        print 'done predicting ',year_num  
        fpr, tpr, thresholds = roc_curve(y_test_cv, 1-cv_pred)
        roc_auc = auc(fpr, tpr)
        print "Area under the ROC curve : %f" % roc_auc
        mean_roc =mean_roc+roc_auc
    print "Area under the ROC curve :", mean_roc/4    
else:      
        pred=np.zeros((x_train.shape[0],4))        
        dtrain_l1 = xgb.DMatrix(x_train.values,missing=999)
 
        listOfModels=range(2007,2014,2)
        modelCnt=0
        for l2_model in listOfModels:
            bst = xgb.Booster() #init model
            bst.load_model('model_'+str(l2_model))
            pred[:,modelCnt]= 1/(1+exp(bst.predict(dtrain_l1)))
            modelCnt=modelCnt+1
            
        dtrain = xgb.DMatrix(pred, label=np.squeeze(y_train.values),missing=999)
        bst = xgb.Booster() #init model
        bst = xgb.train(param, dtrain, num_round)
        bst.save_model('l2model')        


        dtest = xgb.DMatrix(x_test.values,missing=999)
        #L1 predictions
        modelCnt=0 
        pred_test=np.zeros((x_test.shape[0],4))
        for l2_model in listOfModels:
            bst = xgb.Booster() #init model
            bst.load_model('model_'+str(l2_model))
            pred_test[:,modelCnt]= 1/(1+exp(bst.predict(dtest)))
            modelCnt=modelCnt+1
 
        dtest = xgb.DMatrix(pred_test,missing=999)
        bst = xgb.Booster() #init model
        bst.load_model('l2model')
        cv_pred = 1/(1+exp(bst.predict(dtest)))

        fo = csv.writer(submissionFile, lineterminator="\n")
        fo.writerow(["Id","WnvPresent"])
        i = 0
        for item in df_test['Id'].tolist():
            #fo.writerow([item, pred_prob[i,1]])
            fo.writerow([item, 1-cv_pred[i]])
            i += 1
        submissionFile.close()
