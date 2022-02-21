import streamlit as st
import numpy as np
import pandas as pd
import shap
import os
import joblib
import matplotlib.pyplot as plt
from PIL import Image

# page settings
st.set_page_config(page_title="Patient Survival Prediction App", page_icon="🚑", layout="wide")

curr_path = os.path.dirname(os.path.realpath(__file__))

st.markdown("<h1 align='center' class='app_title'><style> .app_title {color: red; word-spacing: 10px; font-size: 70px; margin-top: -100px;}  </style> Patient Survival Prediction App</h1>", unsafe_allow_html=True)

st.image('saved_files/images/title_image.jpg')

st.sidebar.image('saved_files/images/sidebar_image.jpg')

st.markdown("<h2 align='center' class='app_question'><style> .app_question {color: black; word-spacing: 10px; font-size: 50px;} </style>Will the patient survive❓</h2>", unsafe_allow_html=True)

with st.form("Prediction_form"):
    st.subheader("Input the data for prediction:")
    # input elements
    ventilated_apache = st.slider("Ventilated Apache", 0, 1)
    with st.expander("View details for Ventilated Apache"):
        st.write("Ventilated Apache indicates whether the patient was invasively ventilated at the time of the highest scoring arterial blood gas using the oxygenation scoring algorithm, including any mode of positive pressure ventilation delivered through a circuit attached to an endo-tracheal tube or tracheostomy.")
    gcs_verbal_apache = st.slider("GCS Verbal Apache", 1, 5, step=1)
    with st.expander("View details for GCS Verbal Apache"):
        st.write("The verbal component of the Glasgow Coma Scale measured during the first 24 hours which results in the highest APACHE III score.")
    gcs_eyes_apache = st.slider("GCS Eyes Apache", 1, 4, step=1)
    with st.expander("View details for GCS Eyes Apache"):
        st.write("The eye opening component of the Glasgow Coma Scale measured during the first 24 hours which results in the highest APACHE III score")
    gcs_motor_apache = st.slider("gcs_motor_apache", 1, 6, step=1)
    with st.expander("View details for GCS Motor Apache"):
        st.write("The motor component of the Glasgow Coma Scale measured during the first 24 hours which results in the highest APACHE III score")
    intubated_apache = st.slider("Intubated Apache", 0, 1)
    with st.expander("View details for Intubated Apache"):
        st.write("Whether the patient was intubated at the time of the highest scoring arterial blood gas used in the oxygenation score")
    apache_2_bodysystem = st.selectbox("Apache 2 Bodysystem", ['Cardiovascular', 'Respiratory', 'Metabolic', 'Trauma', 'Neurologic', 'Gastrointestinal', 'Renal/Genitourinary', 'Undefined', 'Haematologic'])
    with st.expander("View details for Apache 2 Bodysystem"):
        st.write("Admission diagnosis group for APACHE II")
    apache_3j_bodysystem = st.selectbox("Apache 3J Bodysystem", ['Sepsis', 'Respiratory', 'Metabolic', 'Cardiovascular', 'Trauma', 'Neurological', 'Gastrointestinal', 'Genitourinary', 'Haematological', 'Musculoskeletal/Skin', 'Gynecological'])
    with st.expander("View details for Apache 3J Bodysystem"):
        st.write("Admission diagnosis group for APACHE III")
    apache_2_diagnosis = st.number_input("Apache 2 Diagnosis")
    with st.expander("View details for Apache 2 Diagnosis"):
        st.write("The APACHE II diagnosis for the ICU admission")
    age = st.number_input("Age")
    explain_model = st.checkbox("Show model explanation using Explainable AI")
    # submitting values
    submit_val = st.form_submit_button("Predict")
    
st.sidebar.markdown("<h2 class='sidehead' align='center'><style> .sidehead {color:red; background:white; border-radius: 5px;}</style> Patient Survival Prediction App</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<h2>About the app</h2>", unsafe_allow_html=True)
st.sidebar.write("""
Accurate assessment of mortality risk of the patients at the time of admission
has to be made inorder to determine the medical resources required for the patient.
This app predicts whether a patient admitted to a hospital has a high risk of mortality
or not.
""")
if submit_val:
    # input transformation
    apache_2_bodysystem_dict = {
        'Cardiovascular': 0.4232,
        'Respiratory': 0.1265,
        'Metabolic': 0.0834,
        'Trauma': 0.0418,
        'Neurologic': 0.1297,
        'Gastrointestinal': 0.0984,
        'Renal/Genitourinary': 0.0268,
        'Haematologic': 0.0069,
        'Undefined':  0.0448
    }
    for k,v in apache_2_bodysystem_dict.items():
        if apache_2_bodysystem == k:
            apache_2_bodysystem = v

    apache_3j_bodysystem_dict = {
        'Sepsis': 0.1280,
        'Respiratory': 0.1265,
        'Metabolic': 0.0834,
        'Cardiovascular': 0.3270,
        'Trauma': 0.0418,
        'Neurological': 0.1297,
        'Gastrointestinal': 0.0984,
        'Genitourinary': 0.0236,
        'Haematological': 0.0069,
        'Musculoskeletal/Skin': 0.0127,
        'Gynecological':  0.0034
    }
    for k,v in apache_3j_bodysystem_dict.items():
        if apache_3j_bodysystem == k:
            apache_3j_bodysystem = v

    # creating apache_bodysystem
    apache_bodysystem = 2 * apache_2_bodysystem * apache_3j_bodysystem/(apache_2_bodysystem + apache_3j_bodysystem + 0.00001)

    # list of features
    feats = [
        'ventilated_apache',
        'gcs_motor_apache',
        'gcs_eyes_apache',
        'gcs_verbal_apache', 
        'intubated_apache',
        'apache_bodysystem', 
        'apache_2_diagnosis',
        'age'
        ]
    # saving feature names
    joblib.dump(feats, 'src/models/features.sav')
    # list of corresponding input values
    attribute_vals = [ventilated_apache, gcs_motor_apache, gcs_eyes_apache, gcs_verbal_apache, intubated_apache, apache_bodysystem, apache_2_diagnosis, age]

    # dictionary of features and values
    attr_dict = dict(zip(feats, attribute_vals))
    # dataframe for scaling and model input
    attr_df = pd.DataFrame(attr_dict, index=[1])

    # winsorizing features
    caps = joblib.load('src/utils/winsorizer.sav')
    for item in caps:
        for feat in ['apache_2_diagnosis', 'age']:
            if feat==item[0]:
                attr_df[feat] = np.where(attr_df[feat]<item[1], item[1], np.where(attr_df[feat]>item[2], item[2], attr_df[feat])) 

    # scaling input data
    scaler = joblib.load('src/utils/scaler_selected.sav')
    scaled_df = scaler.transform(attr_df)

    # load the model
    model = joblib.load('src/utils/final_model.sav')

    # predicted value from the model
    value = model.predict(scaled_df)

    # output results
    if value!=0:
        st.markdown("<h1 align='center' class='fail'> <style> .fail {color: red;} </style>Unfortunately, the patient will not survive.</h1>", unsafe_allow_html=True)
    else:
        st.markdown("<h1 align='center' class='success'> <style> .success {color: green;}</style>The patient will survive.</h1>", unsafe_allow_html=True)

    input_attributes = np.array(attribute_vals)
    force_plot_path = '/saved_files/plots/force_plot.png'

    shap_explainer = shap.TreeExplainer(model)
    shap_model_values = shap_explainer.shap_values(input_attributes)
    shap_model_expected_values = shap_explainer.expected_value

    shap.force_plot(shap_model_expected_values[1], 
                    shap_model_values[1], 
                    input_attributes,
                    feats,
                    show=False,
                    matplotlib=True).savefig(curr_path + force_plot_path, bbox_inches='tight')
    
    force_plot_image = Image.open(curr_path + force_plot_path)

if submit_val and explain_model:
    st.markdown('---')
    st.markdown("<h2 align='center'>Marginal contribution of input features in the prediction</h2>", unsafe_allow_html=True)
    st.image(force_plot_image)

    st.markdown("<h2 align='center'>Contribution of features in the model</h2>", unsafe_allow_html=True)
    st.markdown("""<p>
    This plot displays the summary of the impact of top features in a dataset on the model prediction. 
    Each dot represents each individual observation in the dataset. 
    <p style="color:blue;"> Blue dots indicate those observations that impact the prediction that the patient would survive.</p>
    <p style="color:red;"> Red dots indicate those observations that impact the prediction that the patient would not survive.</p> 
    The features are arranged in the decreasing order of priority and hence the topmost feature is
    the most important one.</p>
    """, unsafe_allow_html=True)

    st.image("saved_files/plots/feature_contribution.png")

    st.markdown("""<p>We find that <b>'ventilated_apache'</b> is the feature with the most impact on the model output and <b>'intubated_apache'</b> is the feature with the
    least impact on the model output.</p>
    <p style="color:red;">Higher values of <b>'ventialted_apache'</b> and <b>'age'</b> has a greater impact on the prediction that the patient would not survive. </p>
    <p style="color:green;">Whereas higher values for <b>'gcs_motor_apache'</b>, <b>'gcs_verbal_apache'</b> and <b>'gcs_eyes_apache'</b>
    has a greater impact on the prediction that the patient would survive.</p>""", unsafe_allow_html=True)
st.sidebar.markdown('---')
st.sidebar.markdown("<h3 align='center'><a href='https://github.com/Retinpkumar'>Made by: Retin P Kumar</h3></a>", unsafe_allow_html=True)
