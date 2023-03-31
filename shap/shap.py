
import sklearn
import shap
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

def getContributions(X_test):
    df_x=getDataframe()
    model=getModel()
    explainer = shap.DeepExplainer(model,X_test)
    shap_values = explainer.shap_values(X_test)
    shap_values_mean = pd.DataFrame(shap_values[0], columns=df_x.columns)
    shap_values_mean = shap_values_mean.mean().to_frame()
    # create beeswarm plots for each output
    for i in range(len(shap_values)):
        plt.figure()
        shap.summary_plot(shap_values[i], X_test, plot_type='violin',feature_names=df_x, show=False)
        plt.title('Output {}'.format(i+1))
        plt.show()
    # shap_values_act=shap_values2[0]
    # for i in range(len(shap_values2)):
    #     plt.figure()
    #     shap.summary_plot(shap_values2[i], X_test, plot_type='violin',feature_names=df_x, show=False)
    #     plt.title('Output {}'.format(i+1))
    #     plt.show()
    return shap_values,shap_values_mean

def getDataframe():
    df = pd.read_csv("./PreProcessamentoDados/cleanedData.csv")
    df[['Hour','Minute','Second']] = df.DateTime.str.split(":",expand=True)

    dtypes_dict = {'Hour': float, 'Minute': float}

    # convert the columns to their corresponding data types
    df = df.astype(dtypes_dict)
    #input data
    Predictors= ["Hour","Minute","TemperatureC","DewpointC","PressurehPa","WindDirectionDegrees","WindSpeedKMH","WindSpeedGustKMH","Humidity","HourlyPrecipMM","dailyrainMM","SolarRadiationWatts_m2"]
    # TargetVariable = ["Generated power"]

    df_x=df[Predictors]
    return df_x

def getModel():
    filename = 'Generation Prediction DNN/DNN_finalized_model'
    # load the model from disk
    loaded_model = tf.keras.models.load_model(filename)
    return loaded_model

def execute(X_test):
    #predict ao modelo
    model=getModel()
    predictions = model.predict(X_test)
    all_contr, mean_contr=getContributions(X_test)
    return predictions, all_contr, mean_contr


x_test=[[   0.        ,   15.        ,   11.        ,    8.        ,
        1021.        ,  128.33333333,    8.66666667,   17.        ,
          83.        ,    0.        ,    0.        ,    0.        ],
       [   0.        ,   30.        ,   11.        ,    8.        ,
        1021.        ,  133.33333333,    1.33333333,   19.        ,
          83.        ,    0.        ,    0.        ,    0.        ],
       [   0.        ,   45.        ,   11.        ,    8.        ,
        1021.        ,  127.66666667,    3.        ,   12.33333333,
          82.33333333,    0.        ,    0.        ,    0.        ]]
a, b, c=execute(x_test)
print(a,b,c)
