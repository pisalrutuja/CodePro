##############################################################################
# Import necessary modules
# #############################################################################

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta


###############################################################################
# Define default arguments and create an instance of DAG
# ##############################################################################

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2022,7,30),
    'retries' : 1, 
    'retry_delay' : timedelta(seconds=5)
}


Lead_scoring_inference_dag = DAG(
                dag_id = 'Lead_scoring_inference_pipeline',
                default_args = default_args,
                description = 'Inference pipeline of Lead Scoring system',
                schedule_interval = '@hourly',
                catchup = False
)

load_task = PythonOperator(
            task_id = 'load_task',
            python_callable = load_inferance_data,
            dag = Lead_scoring_inference_dag)
            
###############################################################################
# Create a task for encode_data_task() function with task_id 'encoding_categorical_variables'
# ##############################################################################

encoding_categorical_variables_task = PythonOperator(
                                        task_id = 'encoding_categorical_variables',
                                        python_callable = encode_data_task,
                                        dag = Lead_scoring_inference_dag)

###############################################################################
# Create a task for load_model() function with task_id 'generating_models_prediction'
# ##############################################################################

load_model_task = PythonOperator(
                task_id = 'generating_models_prediction',
                python_callable = load_model,
                dag = Lead_scoring_inference_dag)



###############################################################################
# Create a task for prediction_col_check() function with task_id 'checking_model_prediction_ratio'
# ##############################################################################

checking_model_prediction_ratio_task = PythonOperator(
                                    task_id = 'checking_model_prediction_ratio',
                                    python_callable = prediction_col_check,
                                    dag = Lead_scoring_inference_dag)

###############################################################################
# Create a task for input_features_check() function with task_id 'checking_input_features'
# ##############################################################################

input_features_check_task = PythonOperator(
                        task_id = 'checking_input_features',
                        python_callable = input_features_check,
                        dag = Lead_scoring_inference_dag)

###############################################################################
# Define relation between tasks
# ##############################################################################

load_task >> encoding_categorical_variables_task >> load_model_task >> checking_model_prediction_ratio_task >> input_features_check_task