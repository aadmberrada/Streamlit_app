#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 06:33:47 2021

@author: Abdoul_Aziz_Berrada
"""

import streamlit as st

st. set_page_config(layout="wide", page_icon=":hospital:")
st.set_option('deprecation.showPyplotGlobalUse', False)

import pandas as pd
import numpy as np
import seaborn as sns
import time
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier


tit1, tit2 = st.beta_columns((4,1))

tit1.markdown("<h1 style='text-align: center;'><u>Machine Learning in Healthcare</u> </h1>", unsafe_allow_html=True)


st.sidebar.title("Dataset & Classifier")

dataset_name = st.sidebar.selectbox("Select a Dataset", ("Heart Attack", "Breast Cancer"))

classifier_name = st.sidebar.selectbox("Select a Classifier", ("Logistic Regression", "KNN", "Random Forest", 
                                                                "Decision Tree", "SVM", "Gradient Boosting", "XgBoost"))




from sklearn.preprocessing import LabelEncoder 


LE = LabelEncoder()

def get_dataset(dataset_name):
    
    if dataset_name == "Heart Attack" : 
    
        data = pd.read_csv("https://raw.githubusercontent.com/aadmberrada/Streamit_app/master/heart.csv")
        st.header("Heart Attack Prediction")
        
        return data

    else :
        
        data = pd.read_csv("https://raw.githubusercontent.com/aadmberrada/Streamit_app/master/BreastCancer.csv")
        data["diagnosis"] = LE.fit_transform(data["diagnosis"])
        
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        data["diagnosis"] = pd.to_numeric(data["diagnosis"], errors="coerce")
        
        st.header("Breast Cancer Prediction")
        
        return data

data = get_dataset(dataset_name)


def selected_dataset(dataset_name):
    
    if dataset_name == "Heart Attack":
        
        X = data.drop("output", axis=1)
        
        y = data["output"]
        
        return X, y
    
    elif dataset_name == "Breast Cancer" :
        
        X = data.drop(["id", "diagnosis"], axis=1)
        
        y = data["diagnosis"]
        
        return X, y
    
X, y = selected_dataset(dataset_name)

st.write(data)

st.write("Shape of the Dataset : ", data.shape)

st.write("Number of classes in Y : ", y.nunique())

def plot_op(dataset_name):
    
    plt.figure(figsize=(12, 3))
    
    plt.title("Classes in Y")
    
    if dataset_name == "Heart Attack":
        
        sns.countplot(y)
        
        st.pyplot()
        
    elif dataset_name == "Breast Cancer":
        
        sns.countplot(y)
        
        st.pyplot()
        
plot_op(dataset_name)
    


def add_parameter_ui(clf_name):
    
    params={}

    st.sidebar.write("Select a value")
    
    if clf_name == "Logistic Regression" :
        
        R = st.sidebar.slider("Regularization", 0.1, 10.0, step = 0.1)
        
        MI = st.sidebar.slider("Max_iter", 50, 300, step = 50)
        
        params["R"] = R
        
        params["MI"] = MI

    elif clf_name == 'KNN':
        
        K = st.sidebar.slider("n_neighbors", 1, 20, step = 1)
        
        params["K"] = K
        
    elif clf_name == "SVM":
        
        R = st.sidebar.slider("regularization", 0.1, 10.0, step = 0.1)
        
        kernel = st.sidebar.selectbox("Kernel", ("linear", "poly", "rbf", "sigmoïd", "precomputed"))
        
        params["R"] = R
        
        params["kernel"] = kernel 
        
    elif clf_name == "Random Forest" : 
        
        N = st.sidebar.slider("m_estimators", 50, 500, step = 50)
        
        MD = st.sidebar.slider("max_depth", 2, 20, step = 1)
        
        C = st.sidebar.selectbox("Criterion", ("gini", "entropy"))
        
        params["N"] = N
        
        params["MD"] = MD
        
        params["C"] = C
        
    elif clf_name == "Decision Tree":
        
        MD = st.sidebar.slider("max_depth", 2, 20, step = 1)

        C = st.sidebar.selectbox("criterion", ("gini", "entropy"))
        
        SS = st.sidebar.slider("min_samples_split", 1, 10, step = 1)
        
        params["MD"] = MD
        
        params["C"] = C
        
        params["SS"] = SS
        
    elif clf_name == "Gradient Boosting" : 
        
        N = st.sidebar.slider("l_estimators", 50, 500, step = 50)
        
        LR = st.sidebar.slider("learning rate", 0.01, 0.5, step = 0.01)

        L = st.sidebar.selectbox("Loss", ("deviance", "exponential"))
        
        MD = st.sidebar.slider("max_depth", 2, 20, step = 1)
        

        params["N"] = N
        
        params["M"] = MD
        
        params["LR"] = LR
        
        params["Loss"] = L
        
        
    elif clf_name == "XgBoost" : 
        
        
        N = st.sidebar.slider("N_estimators", 50, 500, step = 50)
        
        LR = st.sidebar.slider("Learning Rate", 0.01, 0.5, step = 0.01)
        
        MD = st.sidebar.slider("Max_depth", 2, 20, step = 1)
        
        O = st.sidebar.selectbox("Objective", ("Binary : Logistic", "Reg : Logistic", "Reg : Squarederror", "reg : gamma"))
        
        G = st.sidebar.slider("Gamma", 0, 10, step = 1)
        
        RL = st.sidebar.slider("Reg_lambda", 1, 5, step = 1)
        
        RA = st.sidebar.slider("Reg_alpha", 0, 5, 1)
        
        CS = st.sidebar.slider("Colsample_bytree", 0.5, 0.1, 1.0)
        
        
        
        params["N"] = N
        
        params["LR"] = LR
        
        params["MD"] = MD
        
        params["0"] = O
        
        params["G"] = G
        
        params["RL"] = RL

        params["RA"] = RA
        
        params["CS"] = CS
        
    RS = st.sidebar.slider("Random State", 0, 100, step = 1)
    
    params["RS"] = RS

    return params

params = add_parameter_ui(classifier_name)




def get_classifier(clf_name, params):
    
    global clf
    
    if clf_name == "Logistic Regression" : 
        
        clf = LogisticRegression(C = params["R"], max_iter= params["MI"])

    elif clf_name == "KNN":
        
        clf = KNeighborsClassifier(n_neighbors = params["K"])

    elif clf_name == "SVM" : 
        
        clf = SVC(kernel = params["kernel"], C = params["R"])
        
    elif clf_name == "Random Forest":
        
        clf = RandomForestClassifier(n_estimators=params["N"],max_depth=params["MD"],criterion=params["C"])

    elif clf_name == "Decision Tree":
        
        clf = DecisionTreeClassifier(max_depth = params["MD"], criterion = params["C"], min_impurity_split = params["SS"])

    elif clf_name == "Gradient Boosting":
        
        clf = GradientBoostingClassifier(n_estimators=params["N"],learning_rate=params["LR"],loss=params["L"],max_depth=params["MD"])

    elif clf_name == "XGBoost":
        
        clf = XGBClassifier(booster="gbtree",n_estimators=params["N"],max_depth=params["MD"],learning_rate=params["LR"],
                            
                            objective=params["O"],gamma=params["G"],reg_alpha=params["A"],reg_lambda=params["L"],colsample_bytree=params["CS"])

    return clf

clf = get_classifier(classifier_name,params)

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support as score

SS = StandardScaler()

def model():
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=65, test_size=0.2)
    
    X_train  = SS.fit_transform(X_train)
    
    X_test = SS.transform(X_test)

    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)

    return y_pred, y_test


y_pred, y_test = model()



st.header(f"1- Prediction for {dataset_name}")
st.subheader(f"Model used : {classifier_name}")
st.subheader(f"Parameters : {params}")


def compute(y_pred, y_test):
    
    
    cm = confusion_matrix(y_test, y_pred)
    
    class_label = ["High Risk", "Low Risk"]
    
    df_cm = pd.DataFrame(cm, index=class_label, columns = ["High Risk", "Low Risk"])
    
    plt.figure(figsize=(12, 7.5))
               
    sns.heatmap(df_cm, annot=True)

    plt.xlabel("Predicted")

    plt.ylabel("True")
    
    plt.title("Confusion Matrix")
    
    st.pyplot()
    
    
    acc = accuracy_score(y_test, y_pred)
    
    precision, recall, f1_score, train_support = score(y_test, y_pred, pos_label=1, average="binary" )
    
    st.subheader("Metrics")
    
    st.text('Precision: {} \nRecall: {} \nF1-Score: {} \nAccuracy: {} %'.format(round(precision, 3), round(recall, 3), round(f1_score,3), round((acc*100),3)))

    
compute(y_pred, y_test)
    

def user_ui(dataset_name, data) : 
    
    user_val = {}
    
    if dataset_name == "Breast Cancer" :
        
        X = data.drop(["id", 'diagnosis'], axis=1)
        
        for col in X.columns:
            
            name = col
            
            value = st.number_input(col, abs(X[col].min()-round(X[col].std())), abs(X[col].max()+round(X[col].std())))
            
            user_val[name] = round((value), 4)
    
    
    elif dataset_name == "Heart Attack":
    
        X = data.drop(["output"], axis=1)
        
        for col in X.columns:
            
            name=col
            
            value = st.number_input(col, abs(X[col].min()-round(X[col].std())), abs(X[col].max()+round(X[col].std())))
            
            user_val[name] = value

    return user_val
    
user_val=user_ui(dataset_name,data)
    
    
    
    
    
def user_predict():
    
    
    global U_pred
    
    if dataset_name == "Breast Cancer":
        
        X = data.drop(["id", 'diagnosis'], axis=1)
        
        U_pred = clf.predict([[user_val[col] for col in X.columns]])
        
    elif dataset_name == "Heart Attack":
       
        X = data.drop(["output"], axis=1)
        
        U_pred = clf.predict([[user_val[col] for col in X.columns]])


    st.subheader("Your Status: ")
    if U_pred == 0:
        st.write(U_pred[0], " - You are not at high risk :)")
    else:
        st.write(U_pred[0], " - You are at high risk :(")
        
        	
user_predict()  #Predict the status of user.


    
    
    
    
    
    
    
    
    
    
    




    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    