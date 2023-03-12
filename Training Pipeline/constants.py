DB_PATH = "/home/Assignment-20220912T150649Z-001/Assignment/02_training_pipeline/notebooks/"
DB_FILE_NAME = "lead_scoring_model_experimentation.db"

DB_FILE_MLFLOW = "Assignment-20220912T150649Z-001/Assignment/02_training_pipeline/notebooks/mlruns"

TRACKING_URI = "http://0.0.0.0:6007"
EXPERIMENT = "Baseline_model_exp01"


# model config imported from pycaret experimentation
model_config = setup(data = data_for_model, target = 'app_complete_flag', 
                   session_id = 42,fix_imbalance=False,
                   ignore_features=['assistance_interaction', 'career_interaction', 'payment_interaction', 'social_interaction', 'syllabus_interaction'],
                   date_features=[],
                   categorical_features = ['city_tier', 'first_platform_c', 'first_utm_medium_c', 'first_utm_source_c'],                             
                   n_jobs=-1,use_gpu=False,
                   log_experiment=True,experiment_name='Baseline_model_exp02',
                   log_plots=True, log_data=True,
                   silent=True, verbose=True,
                   log_profile=False, preprocess=True, normalize = False, transformation = False)

# list of the features that needs to be there in the final encoded dataframe
ONE_HOT_ENCODED_FEATURES = ["interaction_mapping","created_date","city_tier","first_platform_c","first_utm_medium_c","first_utm_source_c","total_leads_droppped","referred_lead","app_complete_flag","assistance_interaction","career_interaction","payment_interaction","social_interaction","syllabus_interaction"]
# list of features that need to be one-hot encoded
FEATURES_TO_ENCODE = ["city_tier","first_platform_c","first_utm_medium_c","first_utm_source_c",'1_on_1_industry_mentorship','call_us_button_clicked',
       'career_assistance', 'career_coach', 'career_impact', 'careers',
       'chat_clicked', 'companies', 'download_button_clicked',
       'download_syllabus', 'emi_partner_click', 'emi_plans_clicked',
       'fee_component_click', 'hiring_partners',
       'homepage_upgrad_support_number_clicked',
       'industry_projects_case_studies', 'live_chat_button_clicked',
       'payment_amount_toggle_mover', 'placement_support',
       'placement_support_banner_tab_clicked', 'program_structure',
       'programme_curriculum', 'programme_faculty',
       'request_callback_on_instant_customer_support_cta_clicked',
       'shorts_entry_click', 'social_referral_click',
       'specialisation_tab_clicked', 'specializations', 'specilization_click',
       'syllabus', 'syllabus_expand', 'syllabus_submodule_expand',
       'tab_career_assistance', 'tab_job_opportunities', 'tab_student_support',
       'view_programs_page', 'whatsapp_chat_click',]
