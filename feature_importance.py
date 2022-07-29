import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

def load_model():
    with open('model.pkl', 'rb') as file:
        data = pickle.load(file)
    file.close()
    return data

data = load_model()

model = data['model']

def show_feature_importance():
    fig, ax = plt.subplots()
    sns.barplot(model.feature_importances_)
    plt.xlabel('importance')
    plt.ylabel('features')
    plt.tight_layout()
    st.pyplot(fig)