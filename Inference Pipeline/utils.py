'''
filename: utils.py
functions: encode_features, load_model
creator: shashank.gupta
version: 1
'''

###############################################################################
# Import necessary modules
# ##############################################################################

import mlflow
import mlflow.sklearn
import pandas as pd

import sqlite3

import os
import logging

from datetime import datetime

###############################################################################
# Define the function to train the model
# ##############################################################################
def load_inferance_data():
    data = pd.read_csv(f"{FILE_PATH}/leadscoring_inference.csv")
    data.drop_duplicates(subset = None, inplace = True)
    data.duplicated().any()
    data.to_csv(f"{FILE_PATH}/leadscoring_inference_data.csv", index =False)

def encode_data_task():
    df = pd.read_csv(f"{FILE_PATH}/leadscoring_inference_data.csv")
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
    df_encoded.to_csv(f"{FILE_PATH}/features.csv",index =False)
    
    
    '''
    This function one hot encodes the categorical features present in our  
    training dataset. This encoding is needed for feeding categorical data 
    to many scikit-learn models.

    INPUTS
        db_file_name : Name of the database file 
        db_path : path where the db file should be
        ONE_HOT_ENCODED_FEATURES : list of the features that needs to be there in the final encoded dataframe
        FEATURES_TO_ENCODE: list of features  from cleaned data that need to be one-hot encoded
        **NOTE : You can modify the encode_featues function used in heart disease's inference
        pipeline for this.

    OUTPUT
        1. Save the encoded features in a table - features

    SAMPLE USAGE
        encode_features()
    '''

###############################################################################
# Define the function to load the model from mlflow model registry
# ##############################################################################

def load_model():
    df = pd.read_csv(f"{FILE_PATH}/features.csv")
    model=joblib.load(f"{FILE_PATH}/features.csv")
    predictions= model.predict(df)
    prediction_df=pd.DataFrame(predictions)
    prediction_df.to_csv(f"{FILE_PATH}/Predictions.csv", index =False)
    '''
    This function loads the model which is in production from mlflow registry and 
    uses it to do prediction on the input dataset. Please note this function will the load
    the latest version of the model present in the production stage. 

    INPUTS
        db_file_name : Name of the database file
        db_path : path where the db file should be
        model from mlflow model registry
        model name: name of the model to be loaded
        stage: stage from which the model needs to be loaded i.e. production


    OUTPUT
        Store the predicted values along with input data into a table

    SAMPLE USAGE
        load_model()
    '''

###############################################################################
# Define the function to check the distribution of output column
# ##############################################################################


    
def prediction_col_check(DB_PATH, DB_FILE_NAME, ML_FLOW_PATH, drfit_db_name):
    cnx_drift = sqlite3.connect(DB_PATH+ drfit_db_name)
    process_flags = pd.read_sql('select * from process_flags', cnx_drift)
    
    if process_flags['Prediction'][0] == 1:
        mlflow.set_tracking_uri("http://0.0.0.0:6007")
        cnx = sqlite3.connect(db_path+db_file_name)
        logged_model = ML_FLOW_PATH
        # Load model as a PyFuncModel.
        loaded_model = mlflow.sklearn.load_model(logged_model)
        # Predict on a Pandas DataFrame.
        X = pd.read_sql('select * from X', cnx)
        predictions_proba = loaded_model.predict_proba(pd.DataFrame(X))
        predictions = loaded_model.predict(pd.DataFrame(X))
        pred_df = X.copy()
        
        pred_df['app_complete_flag'] = predictions
        pred_df[["Prob of Not Churn","Prob of Churn"]] = predictions_proba
        city_tier_mapping = pd.read_sql('select * from city_tier_mapping', cnx)
        pred_df['index_for_map'] = pred_df.index
        final_pred_df = pred_df.merge(city_tier_mapping, on='index_for_map') 
        final_pred_df.to_sql(name='predictions', con=cnx,if_exists='replace',index=False)
        print (pd.DataFrame(predictions_proba,columns=["Prob of Not Churn","Prob of Churn"]).head()) 
        pd.DataFrame(predictions,columns=["Prob of Not Churn","Prob of Churn"]).to_sql(name='Final_Predictions', con=cnx,if_exists='replace',index=False)
        return "Predictions are done and save in Final_Predictions Table"
    else:
        print("Not Required......Skipping")
        
        
    '''
    This function calculates the % of 1 and 0 predicted by the model and  
    and writes it to a file named 'prediction_distribution.txt'.This file 
    should be created in the ~/airflow/dags/Lead_scoring_inference_pipeline 
    folder. 
    This helps us to monitor if there is any drift observed in the predictions 
    from our model at an overall level. This would determine our decision on 
    when to retrain our model.
    

    INPUTS
        db_file_name : Name of the database file
        db_path : path where the db file should be

    OUTPUT
        Write the output of the monitoring check in prediction_distribution.txt with 
        timestamp.

    SAMPLE USAGE
        prediction_col_check()
    '''
###############################################################################
# Define the function to check the columns of input features
# ##############################################################################
   


def input_features_check(ONE_HOT_ENCODED_FEATURES):
    cnx = sqlite3.connect(DB_PATH+DB_FILE_NAME)
    check_table = pd.read_sql(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{ONE_HOT_ENCODED_FEATURES}';", cnx).shape[0]
    if check_table == 1:
        return True
    else:
        return False
    
    '''
    This function checks whether all the input columns are present in our new
    data. This ensures the prediction pipeline doesn't break because of change in
    columns in input data.

    INPUTS
        db_file_name : Name of the database file
        db_path : path where the db file should be
        ONE_HOT_ENCODED_FEATURES: List of all the features which need to be present
        in our input data.

    OUTPUT
        It writes the output in a log file based on whether all the columns are present
        or not.
        1. If all the input columns are present then it logs - 'All the models input are present'
        2. Else it logs 'Some of the models inputs are missing'

    SAMPLE USAGE
        input_col_check()
    '''
   