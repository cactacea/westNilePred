
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

algos = ['bayesianRidge,adaBoost','decisionTree','gradBoost','extraTree','linear']


#All static data structures here
clf  = defaultdict.fromkeys(algos)
master_obj = master()


#All parameters go here.

feat_list = 'all'
learner_id=0 
run_mode = 'CV'
#run_mode = 'submit'
if run_mode == 'CV':
    cv_factor = 0.7
    num_cv_folds =5
else:
    cv_factor = 1
    num_cv_folds =1

#Feature engineering
basepath = 'C:/Users/kpasad/data/ML/projects/westNilePred/'

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

#train_feats = ['month','day','Latitude', 'Longitude']
train_feats = ['year','month','day','Latitude', 'Longitude', 'Tavg', 'Depart', 'DewPoint', 'Heat', 'Cool']
x_train =df_master_train[train_feats]
y_train=df_master_train['WnvPresent'] #Extract the targets
x_test =df_master_test[train_feats]

n_train = df_train.shape[0]
train_sample_idx = range(0,n_train)
num_cv_train_samples = int(df_master_train.shape[0]*cv_factor)
train_sample_idx = deque(range(0,n_train))

#Initialise the classifiers
clf['adaBoost']  = ensemble.AdaBoostRegressor()
clf['decisionTree'] = DecisionTreeRegressor(random_state=0)
clf['gradBoost'] = ensemble.GradientBoostingRegressor(loss='huber',max_depth=2,n_estimators=50)
clf['extraTree'] =ensemble.ExtraTreesRegressor(n_estimators=20)
clf['linear']=linear.LinearRegression( )
clf['log']=linear.LogisticRegression()

clf['bayesianRidge'] = linear.BayesianRidge()
clf['ridge'] = linear.Lasso(alpha=0.1)
clf['randForest']=ensemble.RandomForestRegressor(n_estimators=10, criterion='mse')


master_obj.add_variables(cv_factor=cv_factor,num_cv_folds=num_cv_folds)
algosToTry=['log']
svrParams_C =[100]
svrDegree =[1]
gradBoost_maxD = [2,3,4]
gradBoost_n_est = [50,100,300]

max_predictors = len(x_train.columns)
C=10000

#for max_predictors in [len(x_train.columns)]:
# for C in svrParams_C:
for reg in algosToTry:           
    for cv_fold_idx in range(0,num_cv_folds):
        num_samples_to_shift = int( random.uniform(-1,1)*n_train) #Generate a random, bidirectional, circular shift
        train_sample_idx.rotate(num_samples_to_shift) #Shift the data row indices by this random amount
        rand_train_sample_idx=list(train_sample_idx)  #Convert deque to list
        random.shuffle(rand_train_sample_idx)
        
        cv_train_sample_idx = rand_train_sample_idx[0:num_cv_train_samples]
        cv_test_sample_idx =  rand_train_sample_idx[num_cv_train_samples:n_train]
                  
        
        x_train_cv = x_train.ix[cv_train_sample_idx]
        x_test_cv  =x_train.ix[cv_test_sample_idx]  
                        
        
        y_train_cv = y_train.ix[cv_train_sample_idx]
        y_test_cv  =y_train.ix[cv_test_sample_idx]
        
        
        targetVarError=[]
        print 'fitting'
        clf[reg].fit(x_train_cv,y_train_cv)
        
        if run_mode == 'CV':
            pred = clf[reg].predict_log_proba(x_test_cv)
            pred_prob=1/(1+np.exp(-(pred)))  
            
            print 'done predicting'  
            #Compute ROC curve and area the curve
            fpr, tpr, thresholds = roc_curve(y_test_cv, pred_prob[:,1])
            #fpr, tpr, thresholds = roc_curve(y_test_cv, pred_prob)
            roc_auc = auc(fpr, tpr)
            print "Area under the ROC curve : %f" % roc_auc 
            #n, bins, patches = P.hist(pred_prob,bins=100)         
           
if run_mode == 'submit':
    pred = clf[reg].predict_log_proba(x_test)
    pred_prob=1/(1+np.exp(-(pred)))   
                    
    fo = csv.writer(submissionFile, lineterminator="\n")
    fo.writerow(["Id","WnvPresent"])
    i = 0
    for item in df_test['Id'].tolist():
        fo.writerow([item, pred_prob[i,1]])
        i += 1
    submissionFile.close()
