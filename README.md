# 🌱 Patient Survival Prediction App

![](https://github.com/Retinpkumar/PatientSurvivalPrediction/blob/main/saved_files/images/app_recording.gif)  
[Visit the app](https://agricultural-n2o.herokuapp.com/)

The app predicts whether a patient admitted to a hospital will survive or not.

---


## Motivation



## Model Used
The app uses a machine learning model known as "Random Forest Regressor" for making the prediction. Random Forest belongs to tree based ensemble models category and hence it is tough to interpret the final output of the model. 
Hence, I have used an explainable AI tool known as "SHAP" for understanding the model and interpreting the output.



## Challenges faced
* The dataset had more than 150 features and 150000+ observations
* There were lot of missing values in the dataset.
* The target feature was highly imbalanced and required the usage of sampling techniques.



## Building the app
The app was built using Python3.8 and with the help of Streamlit library.  
Other major libraries used include Numpy, Pandas, Matplotlib, Scikit-learn and Shap.  
The app is currently deployed on heroku.  


## Running the app locally
* Inside a new directory clone the files from the repo using  
<code> git clone https://github.com/Retinpkumar/Agricultural-N2O-Predictor-App.git</code>  
* Create a new virtual environment and activate it.
* Open the terminal and install the libraries and dependancies using  
<code> pip install -r requirements.txt </code>  
* Start and run the app by entering the code given below in the terminal  
<code> streamlit run app.py </code>

<h1 align='center'><b>.&emsp;&ensp;.&emsp;&ensp;.</b></h1>

For further discussions and queries, contact me at: retinpkumar@gmail.com
