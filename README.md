 <h1 align='center'> 🚑 Patient Survival Prediction App </h1>

<div align='center'>
<img src="https://github.com/Retinpkumar/PatientSurvivalPrediction/blob/main/saved_files/images/app_recording.gif">
</div>

<h3 align='justify'> This app predicts whether a patient admitted to a hospital has a high risk of mortality or not. </h3>

<div align='center'>
<a href="https://patient-survival-prediction.herokuapp.com/" align='center'> Visit the app </a>
</div>
 
<h2> 📖 Table of Contents </h2>
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#motivation"> ➤ Motivation </a></li>
    <li><a href="#structure"> ➤ Folder Structure </a></li>
    <li><a href="#model_used"> ➤ Model used </a></li>
    <li><a href="#challenges"> ➤ Challenges faced </a></li>
    <li><a href="#build"> ➤ Building the app </a></li>
    <li><a href="#run"> ➤ Running the app locally </a></li>
    
  </ol>
</details>

![---](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)


<h2 id="motivation" > 🎯 Motivation </h2>

Accurate assessment of mortality risk of the patients at the time of admission has to be made inorder to determine and make available, the medical resources required for the patient.

  Rather than going the conventional way of checking vital signs of the patients, details regarding quickly repeatable and efficient laboratory tests have been used as the features for building the model.  
  
  Due to the quick repetitive nature of the tests, these features can be used for generating a quick assesment of the mortality condition of the patient. Leveraging these sets of features, this app determines whether a patient admitted to a hospital has a high risk of mortality
or not.

![---](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)


<h2 id='structure'> 📂 Folder Structure </h2>


```
PatientSurvivalPrediction
|
|—— .streamlit
|—— saved_files
|    |—— data
|    |    |—— Data Dictionary.csv
|    |    |—— Dataset.csv
|    |
|    |—— images
|    |    |—— app_recording.gif
|    |    |—— sidebar_image.jpg
|    |    |—— title_image.jpg
|    |
|    |—— notebooks
|    |    |—— Patient_survival_prediction.ipynb
|    |
|    |—— plots
|         |—— feature_contribution.png
|         |—— force_plot.png
|
|—— src
|    |—— models
|    |    |—— features.sav
|    |
|    |—— utils
|        |—— final_model.sav
|        |—— scaler_selected.sav
|        |—— shap_values.sav
|        |—— winsorizer.sav
|
|—— .gitignore
|—— Procfile
|—— README.md
|—— app.py
|—— requirements.txt
|—— runtime.txt
|—— setup.sh
```

![---](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)

<h2 id="model_used"> 💻 Model Used </h2>

The app uses a machine learning model known as "Random Forest Regressor" for making the prediction. Random Forest belongs to tree based ensemble models category and hence it is tough to interpret the final output of the model.  

Hence, I have used an explainable AI tool known as "SHAP" for understanding the model and interpreting the output.  

<img src="https://github.com/Retinpkumar/PatientSurvivalPrediction/blob/main/saved_files/plots/feature_contribution.png">

Check the app for more detailed explanation of the model and the predicted output.

![---](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)

<h2 id="challenges"> 🧩 Challenges faced </h2>

* The dataset had 186 features and most of them were correlated.  

* There were lot of missing values in the dataset and majority of the features had more than 50% missing values.  

* The target feature was highly imbalanced with data corresponding to hospital deaths constituting only about 8.63% of the entire data. This required the usage of sampling techniques.  

* The dataset had lots of outliers as well.

![---](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)

<h2 id="build"> 📚 Building the app </h2>

The app was built using Python3.8 and with the help of Streamlit library.  
Other major libraries used include:
<ul>
  <li>Numpy</li> 
  <li>Pandas</li>
  <li>Matplotlib</li>
  <li>Scikit-learn</li>
  <li>Shap</li>
</ul>
The app is currently deployed on Heroku.  

![---](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)

<h2 id="run"> 💾 Running the app locally </h2>

* Inside a new directory clone the files from the repo using  
<code> https://github.com/Retinpkumar/PatientSurvivalPrediction</code>  

* Create a new virtual environment and activate it.

* Open the terminal and install the libraries and dependancies using  
<code> pip install -r requirements.txt </code>  

* Start and run the app by entering the code given below in the terminal  
<code> streamlit run app.py </code>

![---](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)

For further discussions and queries, contact me at: retinpkumar@gmail.com
