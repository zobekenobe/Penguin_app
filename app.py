import streamlit as st
import pickle
from predict_page import show_predict_page
from feature_importance import show_feature_importance
from about import about_page


page = st.sidebar.selectbox(label = "Select the page", options = ['About', 'Predict', 'Feature Importance'])

if page == 'About':
    about_page()
elif page == 'Predict':
    show_predict_page()
else:
    show_feature_importance()
