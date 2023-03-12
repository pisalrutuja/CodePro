###############################################################################
# Import necessary modules
# ##############################################################################

import pandas as pd
import numpy as np

import sqlite3
from sqlite3 import Error

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from scripts.constants import *


###############################################################################
# Define the function to encode features
# ##############################################################################

def load_inferance_data():
    data = pd.read_csv(f"{FILE_PATH}/leadscoring_inference.csv")
    data.drop_duplicates(subset = None, inplace = True)
    data.duplicated().any()
    return data




def encode_features(DB_FILE_NAME, DB_PATH, ONE_HOT_ENCODED_FEATURES, FEATURES_TO_ENCODE):
        df = pd.read_csv(f"{DB_PATH}/DB_FILE_NAME")
    df_encoded=pd.DataFrame(columns = ONE_HOT_ENCODED_FEATURES)
    df_placeholder=pd.DataFrame()
    for f in FEATURES_TO_ENCODE:
        if (f in df.columns):
            encoded = pd.get_dummies(df[f])
            encoded = encoded.add.prefix(f+'_')
            df_placeholder=pd.concat([df_placeholder,encoded], axis = 1)
        else:
            print('Feature Not Found')
            return df
    for f in df_encoded.columns:
        if f in df.columns:
            df_encoded[f]=df[f]
        if f in df_placeholder.columns:
            df_encoded[f]=df_placeholder[f]
    df_encoded.fillna(0, inplace = True)
    features = df_encoded[df_encoded.columns.drop("target")]
    target = df_encoded["target"]
    return features, target
   

'''
    This function one hot encodes the categorical features present in our  
    training dataset. This encoding is needed for feeding categorical data 
    to many scikit-learn models.

    INPUTS
        db_file_name : Name of the database file 
        db_path : path where the db file should be
        ONE_HOT_ENCODED_FEATURES : list of the features that needs to be there in the final encoded dataframe
        FEATURES_TO_ENCODE: list of features  from cleaned data that need to be one-hot encoded
       

    OUTPUT
        1. Save the encoded features in a table - features
        2. Save the target variable in a separate table - target


    SAMPLE USAGE
        encode_features()
        
    **NOTE : You can modify the encode_featues function used in heart disease's inference
        pipeline from the pre-requisite module for this.
    '''


###############################################################################
# Define the function to train the model
# ##############################################################################

def get_train_model(DB_PATH,DB_FILE_NAME,drfit_db_name):
    cnx_drift = sqlite3.connect(DB_PATH+drfit_db_name)
    process_flags = pd.read_sql('select * from process_flags', cnx_drift)
    
    if process_flags['Data_Preparation'][0] == 1:
        cnx = sqlite3.connect(DB_PATH+DB_FILE_NAME)
        X = pd.read_sql('select * from X', cnx)
        y = pd.read_sql('select * from y', cnx)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

        model_config = {
        'boosting_type': 'gbdt',
        'class_weight': None,
        'colsample_bytree': 1.0,
        'importance_type': 'split' ,
        'learning_rate': 0.1,
        'max_depth': -1,
        'min_child_samples': 20,
        'min_child_weight': 0.001,
        'min_split_gain': 0.0,
        'n_estimators': 100,
        'n_jobs': -1,
        'num_leaves': 31,
        'objective': None,
        'random_state': 42,
        'reg_alpha': 0.0,
        'reg_lambda': 0.0,
        'silent': 'warn',
        'subsample': 1.0,
        'subsample_for_bin': 200000 ,
        'subsample_freq': 0
        }


        #Model Training

        with mlflow.start_run(run_name='run_LightGB_withoutHPTune') as run:
            #Model Training
            clf = lgb.LGBMClassifier()
            clf.set_params(**model_config) 
            clf.fit(X_train, y_train)

            mlflow.sklearn.log_model(sk_model=clf,artifact_path="models", registered_model_name='LightGBM')
            mlflow.log_params(model_config)    

            # predict the results on training dataset
            y_pred=clf.predict(X_test)

            # # view accuracy
            # acc=accuracy_score(y_pred, y_test)
            # conf_mat = confusion_matrix(y_pred, y_test)
            # mlflow.log_metric('test_accuracy', acc)
            # mlflow.log_metric('confustion matrix', conf_mat)
            
            
            #Log metrics
            acc=accuracy_score(y_pred, y_test)
            conf_mat = confusion_matrix(y_pred, y_test)
            precision = precision_score(y_pred, y_test,average= 'macro')
            recall = recall_score(y_pred, y_test, average= 'macro')
            f1 = f1_score(y_pred, y_test, average='macro')
            cm = confusion_matrix(y_test, y_pred)
            tn = cm[0][0]
            fn = cm[1][0]
            tp = cm[1][1]
            fp = cm[0][1]
            class_zero = precision_recall_fscore_support(y_test, y_pred, average='binary',pos_label=0)
            class_one = precision_recall_fscore_support(y_test, y_pred, average='binary',pos_label=1)

            mlflow.log_metric('test_accuracy', acc)
            mlflow.log_metric("f1", f1)
            mlflow.log_metric("Precision", precision)
            mlflow.log_metric("Recall", recall)
            mlflow.log_metric("Precision_0", class_zero[0])
            mlflow.log_metric("Precision_1", class_one[0])
            mlflow.log_metric("Recall_0", class_zero[1])
            mlflow.log_metric("Recall_1", class_one[1])
            mlflow.log_metric("f1_0", class_zero[2])
            mlflow.log_metric("f1_1", class_one[2])
            mlflow.log_metric("False Negative", fn)
            mlflow.log_metric("True Negative", tn)
            # mlflow.log_metric("f1", f1_score)

            runID = run.info.run_uuid
            print("Inside MLflow Run with id {}".format(runID))
    else:
        print("Not Required......Skipping")

    '''
    This function setups mlflow experiment to track the run of the training pipeline. It 
    also trains the model based on the features created in the previous function and 
    logs the train model into mlflow model registry for prediction. The input dataset is split
    into train and test data and the auc score calculated on the test data and
    recorded as a metric in mlflow run.   

    INPUTS
        db_file_name : Name of the database file
        db_path : path where the db file should be


    OUTPUT
        Tracks the run in experiment named 'Lead_Scoring_Training_Pipeline'
        Logs the trained model into mlflow model registry with name 'LightGBM'
        Logs the metrics and parameters into mlflow run
        Calculate auc from the test data and log into mlflow run  

    SAMPLE USAGE
        get_trained_model()
    '''

def get_validation_unseen_set(features, validation_frac=0.05,sample = False, sample_frac=0.1):
    if not sample:
        dataset = features.copy()
    else:
        dataset = features.sample(frac=sample_frac)
    data = dataset.sample(frac=(1-validation_frac), random_state=786)
    data_unseen = dataset.drop(data.index)
    data.reset_index(inplace=True, drop=True)
    data_unseen.reset_index(inplace=True, drop=True)
    return data,data_unseen
                                   