# ❤ Patient Survival Prediction App

![](https://github.com/Retinpkumar/PatientSurvivalPrediction/blob/main/saved_files/images/app_recording.gif)  
[Visit the app](https://patient-survival-prediction.herokuapp.com/)

This app predicts whether a patient admitted to a hospital has a high risk of mortality or not.

---


## Motivation
Accurate assessment of mortality risk of the patients at the time of admission has to be made inorder to determine and make available, the medical resources required for the patient.  

Rather than going the conventional way of checking vital signs of the patients, details regarding quickly repeatable and efficient laboratory tests have been used as the features for building the model. Due to the quick repetitive nature of the tests, these features can be used for generating a quick assesment of the mortality condition of the patient.  

Leveraging these sets of features, this app determines whether a patient admitted to a hospital has a high risk of mortality
or not.


## Model Used
The app uses a machine learning model known as "Random Forest Regressor" for making the prediction. Random Forest belongs to tree based ensemble models category and hence it is tough to interpret the final output of the model. 
Hence, I have used an explainable AI tool known as "SHAP" for understanding the model and interpreting the output.



## Challenges faced
* The dataset had 186 features and most of them were correlated.
* There were lot of missing values in the dataset and majority of the features had more than 50% missing values.
* The target feature was highly imbalanced with data corresponding to hospital deaths constituting only about 8.63% of the entire data. This required the usage of sampling techniques.
* The dataset had lots of outliers as well.


## Building the app
The app was built using Python3.8 and with the help of Streamlit library.  
Other major libraries used include Numpy, Pandas, Matplotlib, Scikit-learn and Shap.  
The app is currently deployed on heroku.  


## Running the app locally
* Inside a new directory clone the files from the repo using  
<code> https://github.com/Retinpkumar/PatientSurvivalPrediction</code>  
* Create a new virtual environment and activate it.
* Open the terminal and install the libraries and dependancies using  
<code> pip install -r requirements.txt </code>  
* Start and run the app by entering the code given below in the terminal  
<code> streamlit run app.py </code>

<h1 align='center'><b>.&emsp;&ensp;.&emsp;&ensp;.</b></h1>

For further discussions and queries, contact me at: retinpkumar@gmail.com
