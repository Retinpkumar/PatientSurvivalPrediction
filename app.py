import streamlit as st
from src.model import predict
from src.transformer import input_transformer
import numpy as np
import pandas as pd
# page settings
st.set_page_config(page_title="Patient Survival Prediction App",
                   page_icon="🏥",
                   layout="centered")
# page header
st.title(f"Patient Survival Prediction App")


with st.form("Prediction_form"):
    # form header
    st.header("Enter the input factors:")
    # input elements
    apache_3j_bodysystem = st.selectbox("apache_3j_bodysystem: ",
                                        ["Cardiovascular", "Neurological", "Sepsis", "Respiratory", "Gastrointestinal",
                                         "Metabolic", "Trauma", "Genitourinary", "Musculoskeletal/Skin", "Hematological",
                                         "Gynecological"])
    apache_4a_hospital_death_prob = st.number_input("apache_4a_hospital_death_prob: ")
    gcs_verbal_apache = st.number_input("gcs_verbal_apache: ")
    apache_2_bodysystem = st.selectbox("apache_2_bodysystem: " ,["Cardiovascular", "Neurologic","Respiratory", "Gastrointestinal","Metabolic", "Trauma","Undefined diagnoses","Renal/Genitourinary", "Haematologic","Undefined Diagnoses"])
    ventilated_apache = st.selectbox("ventilated_apache: ",[0,1])

    # submitting values
    submit_val = st.form_submit_button("Predict")

if submit_val:
    # for apache_3j_bodysystem
    if apache_3j_bodysystem == "Cardiovascular":
        apache_3j_bodysystem = 1
    elif apache_3j_bodysystem == "Neurological":
        apache_3j_bodysystem = 2
    elif apache_3j_bodysystem == "Sepsis":
        apache_3j_bodysystem = 3
    elif apache_3j_bodysystem == "Respiratory":
        apache_3j_bodysystem = 4
    elif apache_3j_bodysystem == "Gastrointestinal":
        apache_3j_bodysystem = 5
    elif apache_3j_bodysystem == "Metabolic":
        apache_3j_bodysystem = 6
    elif apache_3j_bodysystem == "Trauma":
        apache_3j_bodysystem = 7
    elif apache_3j_bodysystem == "Genitourinary":
        apache_3j_bodysystem = 8
    elif apache_3j_bodysystem == "Musculoskeletal/Skin":
        apache_3j_bodysystem = 9
    elif apache_3j_bodysystem == "Hematological":
        apache_3j_bodysystem = 10
    else:
        apache_3j_bodysystem = 11

    # for apache_2_bodysystem
    if apache_2_bodysystem == "Cardiovascular":
        apache_2_bodysystem = 1
    elif apache_2_bodysystem == "Neurological":
        apache_2_bodysystem = 2
    elif apache_2_bodysystem == "Respiratory":
        apache_2_bodysystem = 3
    elif apache_2_bodysystem == "Gastrointestinal":
        apache_2_bodysystem = 4
    elif apache_2_bodysystem == "Metabolic":
        apache_2_bodysystem = 5
    elif apache_2_bodysystem == "Trauma":
        apache_2_bodysystem = 6
    elif apache_2_bodysystem == "Undefined diagnoses":
        apache_2_bodysystem = 7
    elif apache_2_bodysystem == "Renal/Genitourinary":
        apache_2_bodysystem = 8
    elif apache_2_bodysystem == "Hematological":
        apache_2_bodysystem = 9
    elif apache_2_bodysystem == "Undefined Diagnoses":
        apache_2_bodysystem = 10
    else:
        apache_2_bodysystem = 11

    # for apache_4a_hospital_death_prob
    if apache_4a_hospital_death_prob < 0.05:
        apache_4a_hospital_death_prob = 0
    else:
        apache_4a_hospital_death_prob = 1

    # list of features
    feats = ['apache_3j_bodysystem', 'apache_4a_hospital_death_prob', 'gcs_verbal_apache', 'apache_2_bodysystem', 'ventilated_apache']
    # list of corresponding input values
    attribute_vals = [apache_3j_bodysystem, apache_4a_hospital_death_prob, gcs_verbal_apache, apache_2_bodysystem, ventilated_apache]
    # dictionary of features and values
    attr_dict = dict(zip(feats, attribute_vals))
    # dataframe for scaling and model input
    attr_df = pd.DataFrame(attr_dict, index=[1])
    # getting values from training data to transform incoming inputs
    train_df = input_transformer()
    # log transform
    attr_df['gcs_verbal_apache'] = np.log1p(attr_df['gcs_verbal_apache'])
    # capping upper and lower values
    attr_df['gcs_verbal_apache'] = np.where(attr_df['gcs_verbal_apache']>max(train_df['gcs_verbal_apache']), max(train_df['gcs_verbal_apache']),
                        np.where(attr_df['gcs_verbal_apache']<min(train_df['gcs_verbal_apache']), min(train_df['gcs_verbal_apache']), attr_df['gcs_verbal_apache']))

    # predicted value from the model
    value = predict(attributes=attr_df)
    # results header
    st.header("Result: ")
    # output results
    if value:
        st.error(f"The patient died during hospitalization")
    else:
        st.success(f"The patient survived during hospitalization")
        st.balloons()
