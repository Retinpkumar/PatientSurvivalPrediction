# 🌱 Agricultural N<sub>2</sub>O flux Predictor App

![](https://github.com/Retinpkumar/Agricultural-N2O-Predictor-App/blob/main/images/app_recording.gif)  
[Visit the app](https://agricultural-n2o.herokuapp.com/)

The app predicts the amount of Agricultural Nitrous Oxide flux released into the atmosphere from intensively managed cropping systems.

---


## Motivation
Nitrous Oxide is one of the potent greenhouse gas that is accumulating in the atmosphere at extraordinary rates. This can be largely attributed to intensification of agricultural cropping systems where the cultivated soil contribute almost 60% of the agricultural flux.

Though there were previous attempts at identifying and quantifying the agricultural Nitrous Oxide flux, these attempts could only predict the results with an R2-score ranging from 25-40%. The model used in this app is aimed at outperforming the previous models and has been able to succesfully produce the results with almost 78% R2-score.



## Model Used
The app uses a machine learning model known as "Random Forest Regressor" for making the prediction. Random Forest belongs to tree based ensemble models category and hence it is tough to interpret the final output of the model. 
Hence, I have used an explainable AI tool known as "SHAP" for understanding the model and interpreting the output.



## Challenges faced
* The data that quantified Nitrous Oxide flux was heavily right skewed and also contained few negative values.  
* The data for Water frontage pore space, Ammonia and Nitrate had some values missing.  
* Vegetation data was highly imbalanced with almost 84% of the data were recorded for Corn. Data for Wheat and Soy constituted only about 8.5 and 7.5% each.



## Building the app
The app was built using Python3.8 and with the help of Streamlit library.  
Other major libraries used include Numpy, Pandas, Matplotlib, Scikit-learn and Shap.  
The app is currently deployed on heroku.  


## High level overview
![](https://github.com/Retinpkumar/Agricultural-N2O-Predictor-App/blob/main/images/app_working.png)

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
