# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 14:04:03 2021

@author: YG
"""

import shap
import streamlit as st
import streamlit.components.v1 as components
import xgboost
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


# Disable warning PyplotGlobalUseWarning
st.set_option('deprecation.showPyplotGlobalUse', False)


@st.cache

def load_data(display):
    return shap.datasets.boston(display=display)

# function to show force plot
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

# title
st.title("XAI with SHAP")

# define application areas of SHAP on sidebar
area = st.sidebar.radio('Application Area:',['Number','Text','Image'])

############----------------------Number------------------############
if area == 'Number':
    clf = st.sidebar.selectbox('Choose Classifier:',['XGBoost','Random Forest','Decision Tree'])

    # Load dataset
    X, y = load_data(display=False)
    X_display, y_display = load_data(display=True)
    # create a train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

    ##----------------------XGBoost
    if clf == 'XGBoost':
     
        d_train = xgboost.DMatrix(X_train, label=y_train)
        d_test = xgboost.DMatrix(X_test, label=y_test)

        # train XGBoost model
        model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=y), 100) 

    ##-----------------RF
    if clf == 'Random Forest':

        ## Random Foreast
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor()
        model.fit(X_train, y_train)

    ##------------------Text
    if clf == 'Decision Tree':
        
        ## Decision Tree
        from sklearn.tree import DecisionTreeRegressor
        model = DecisionTreeRegressor(random_state = 0)
        model.fit(X_train, y_train)

    ##----------
    # explain the model's predictions using SHAP
    # (same syntax works for LightGBM, CatBoost, scikit-learn and spark models)
    explainer = shap.Explainer(model)
    shap_values = explainer.shap_values(X)

    # Explain dataset's features (dataset dictionary)
    # TODO poner mas features
    dataset_dictionary = pd.DataFrame({
        'Feature': ["CRIM",
                    "ZN",
                    "INDUS",
                    "CHAS",
                    "NOX",
                    "...more"],
        'Description': ["per capita crime rate by town",
                        "proportion of residential land zoned for lots over 25,000 sq.ft.",
                        "proportion of non-retail business acres per town",
                        "Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)",
                        "nitric oxides concentration (parts per 10 million)",
                        "...more"]
    })
    st.write("Dataset dictionary:")
    st.write(dataset_dictionary)
    st.write("Source: https://scikit-learn.org/stable/datasets/toy_dataset.html#boston-dataset")

    # visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
    selected_index = st.selectbox('Index', (list(range(10))))
    st_shap(shap.force_plot(explainer.expected_value, shap_values[selected_index, :], X.iloc[selected_index, :]))

    # visualize the training set predictions
    selected_num = st.select_slider('number of examples', options=list(np.arange(10, 100, 10)))
    st_shap(shap.force_plot(explainer.expected_value, shap_values[:selected_num, :], X.iloc[:selected_num, :]), 400)

    # visualize the summary plot
    # st.pyplot(shap.summary_plot(shap_values, X)) another way to show the summary plot
    plt.title('Feature importance based on SHAP values')
    shap.summary_plot(shap_values, X, show=False)
    st.pyplot()

    # visualize Bar chart of mean importance
    plt.title('Bar chart of mean importance')
    shap.summary_plot(shap_values, X_display, plot_type="bar")
    st.pyplot()

    # visualize SHAP for each feature
    st.title("SHAP Dependence Plots")
    selected_feature = st.selectbox('Feature', options=X.columns)
    shap.dependence_plot(selected_feature, shap_values, X, display_features=X_display)
    st.pyplot()
#for name in X.columns:
    #shap.dependence_plot(name, shap_values, X, display_features=X_display)
    #st.pyplot()
#streamlit run C:/Users/YG/Documents/GitHub/xai-shap-dashboard/main.py

##########-----------------Text---------------############

if area == 'Text':
    # Emotion Classification
    import transformers
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import scipy as sp
    from datasets import load_dataset
    import torch

    # load data
    dataset  = load_dataset("emotion", split = "train")
    data = pd.DataFrame({'text':dataset['text'],'emotion':dataset['label']})

    # tokenize
    tokenizer = AutoTokenizer.from_pretrained("nateraw/bert-base-uncased-emotion",use_fast=True)
    model_text = AutoModelForSequenceClassification.from_pretrained("nateraw/bert-base-uncased-emotion")

     # set mapping between label and id
    id2label = model_text.config.id2label
    label2id = model_text.config.label2id
    labels = sorted(label2id, key=label2id.get)

    def f(x):
        tv = torch.tensor([tokenizer.encode(v, padding='max_length', max_length=128,truncation=True) for v in x])
        attention_mask = (tv!=0).type(torch.int64)
        outputs = model_text(tv,attention_mask=attention_mask)[0].detach().cpu().numpy()
        scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
        val = sp.special.logit(scores)
        return val

    # explainer
    e_text = shap.Explainer(f,tokenizer,output_names=labels)
    shap_values_text = e_text(data['text'][0:10])

    # plot
    shap.plots.bar(shap_values_text[:,:,"joy"].mean(0))
    st.pyplot()

##########-----------------Image-------------##############

if area == 'Image':
    # fashion mnist classification with neural network
    import tensorflow as tf
    from tensorflow import keras
    import warnings
    warnings.filterwarnings('ignore')

    ## load data
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (X_train_img, y_train_img), (X_test_img, y_test_img) = fashion_mnist.load_data()[:1000]
    X_train_img, X_valid_img, y_train_img, y_valid_img = train_test_split(X_train_img, y_train_img, test_size=0.1, random_state=1)

    # scaling data to range 0-1
    X_train_img = X_train_img / 255
    X_valid_img = X_valid_img / 255
    X_test_img = X_test_img / 255

    # define class names
    class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                   "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

    # define model
    model_img = keras.models.Sequential([
        keras.layers.Flatten(input_shape = [28, 28]),
        keras.layers.Dense(100, activation = 'relu'),
        keras.layers.Dense(50, activation = 'relu'),
        keras.layers.Dense(20, activation = 'relu'),
        keras.layers.Dense(10, activation = 'softmax')
    ])

    # compiling model

    model_img.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(),
                  optimizer = tf.keras.optimizers.Adam(), metrics = ['accuracy'])

    # fitting model

    model_history = model_img.fit(X_train_img, y_train_img, validation_data = (X_valid_img, y_valid_img), epochs = 30)

    # select a set of background examples to take an expectation over
    background = X_train_img[np.random.choice(X_train_img.shape[0], 100, replace=False)]

    # explain predictions of the model on three images
    e = shap.DeepExplainer(model_img, background)
    # ...or pass tensors directly
    # e = shap.DeepExplainer((model.layers[0].input, model.layers[-1].output), background)
    shap_values_img = e.shap_values(X_test_img[1:5])

    # plot the feature attributions
    shap.image_plot(shap_values_img, -X_test_img[1:5])
    st.pyplot()

