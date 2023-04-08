import sklearn
import shap
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def getshapvalues(X_test):
    df_x = getDataframeColumns()
    model = getModel()
    explainer = shap.DeepExplainer(model, X_test)
    shap_values = explainer.shap_values(X_test)
    shap_values_mean = pd.DataFrame(shap_values[0], columns=df_x)
    shap_values_mean = shap_values_mean.mean().to_frame()
    # create beeswarm plots for each output
    # for i in range(len(shap_values)):
    #     plt.figure()
    #     shap.summary_plot(shap_values[i], X_test, plot_type='violin', feature_names=df_x, show=False)
    #     plt.title('Output {}'.format(i + 1))
    #     plt.show()
    return shap_values, shap_values_mean


def readDataframe():
    try:
        df = pd.read_csv("./PreProcessamentoDados/cleanedData.csv")
        df[['Hour', 'Minute', 'Second']] = df.DateTime.str.split(":", expand=True)
    except:
        df = pd.read_csv("../PreProcessamentoDados/cleanedData.csv")
        df[['Hour', 'Minute', 'Second']] = df.DateTime.str.split(":", expand=True)

    dtypes_dict = {'Hour': float, 'Minute': float}

    # convert the columns to their corresponding data types
    df = df.astype(dtypes_dict)
    return df


def getDataframeColumns():
    df = readDataframe()
    # input data
    predictors = ["Hour", "Minute", "TemperatureC", "DewpointC", "PressurehPa", "WindDirectionDegrees", "WindSpeedKMH",
                  "WindSpeedGustKMH", "Humidity", "HourlyPrecipMM", "dailyrainMM", "SolarRadiationWatts_m2"]
    # TargetVariable = ["Generated power"]

    df_x = df[predictors]
    return df_x.columns


def getRandomX_test():
    df = readDataframe()
    TargetVariable = ["Generated power"]
    Predictors = ["Hour", "Minute", "TemperatureC", "DewpointC", "PressurehPa", "WindDirectionDegrees", "WindSpeedKMH",
                  "WindSpeedGustKMH", "Humidity", "HourlyPrecipMM", "dailyrainMM", "SolarRadiationWatts_m2"]
    X = df[Predictors].values
    y = df[TargetVariable].values

    PredictorScaler = StandardScaler()
    TargetVarScaler = StandardScaler()

    # Storing the fit object for later reference
    PredictorScalerFit = PredictorScaler.fit(X)
    TargetVarScalerFit = TargetVarScaler.fit(y)
    X = PredictorScalerFit.transform(X)
    y = TargetVarScalerFit.transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_test, TargetVarScalerFit



def getModel():
    try:
        filename = 'Generation Prediction DNN/DNN_model.h5'
        # load the model from disk
        loaded_model = tf.keras.models.load_model(filename)
    except:
        filename = '../Generation Prediction DNN/DNN_model.h5'
        # load the model from disk
        loaded_model = tf.keras.models.load_model(filename)
    
    return loaded_model

def fix_xtest(x_test, x_train, y_train):
    PredictorScaler = StandardScaler()
    TargetVarScaler = StandardScaler()

    # Storing the fit object for later reference
    PredictorScalerFit = PredictorScaler.fit(x_train)
    X = PredictorScalerFit.transform(x_test)
    
    TargetVarScalerFit = TargetVarScaler.fit(y_train)
    return X,TargetVarScalerFit

def execute(x_test,PredictorScalerFit):
    model = getModel()
    predictions = model.predict(x_test)
    predictions=PredictorScalerFit.inverse_transform(predictions)
    all_contrib, mean_contrib = getshapvalues(x_test)
    return predictions, all_contrib, mean_contrib


def testingExecute():
    x_test,TargetVarScalerFit = getRandomX_test()
    prediction, all_contribution, mean_contribution = execute(x_test,TargetVarScalerFit)
    return prediction, all_contribution, mean_contribution

def fixedXtest():
    # predict ao modelo
    x_test = [[0., 15., 11., 8.,
               1021., 128.33333333, 8.66666667, 17.,
               83., 0., 0., 0.],
              [0., 30., 11., 8.,
               1021., 133.33333333, 1.33333333, 19.,
               83., 0., 0., 0.],
              [0., 45., 11., 8.,
               1021., 127.66666667, 3., 12.33333333,
               82.33333333, 0., 0., 0.]]
    x_test = np.array(x_test)
    x_train = np.load('x_train.npy')
    y_train = np.load('y_train.npy')
    x_test,TargetVarScalerFit = fix_xtest(x_test, x_train, y_train)
    prediction, all_contribution, mean_contribution = execute(x_test,TargetVarScalerFit)
    return prediction, all_contribution, mean_contribution

def getContributions(data):
    if not isinstance(data, np.array):
        data = np.array(data)
    #ler o dataset
    #x_train sample do df => guardado num file
    #y_train sample do df => guardado num file 
    #fit e tranform a essas variáveis
    x_train = np.load('x_train.npy')
    y_train = np.load('y_train.npy')
    x_test,TargetVarScalerFit = fix_xtest(data, x_train, y_train)
    prediction, all_contribution, mean_contribution = execute(x_test,TargetVarScalerFit)
    return prediction, all_contribution, mean_contribution

# prediction, all_contribution, mean_contribution = testingExecute()
# print('pre\n', prediction)
# print('all_contributions\n', all_contribution)
# print('mean_contribution\n', mean_contribution)

def saveTrains():
    df = readDataframe()
    TargetVariable = ["Generated power"]
    Predictors = ["Hour", "Minute", "TemperatureC", "DewpointC", "PressurehPa", "WindDirectionDegrees", "WindSpeedKMH",
                  "WindSpeedGustKMH", "Humidity", "HourlyPrecipMM", "dailyrainMM", "SolarRadiationWatts_m2"]
    X = df[Predictors].values
    y = df[TargetVariable].values

    PredictorScaler = StandardScaler()
    TargetVarScaler = StandardScaler()

    # Storing the fit object for later reference
    PredictorScalerFit = PredictorScaler.fit(X)
    TargetVarScalerFit = TargetVarScaler.fit(y)
    X = PredictorScalerFit.transform(X)
    y = TargetVarScalerFit.transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    np.save('x_train.npy', X_train)
    np.save('y_train.npy', y_train)
