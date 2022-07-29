import streamlit as st
import pickle
import numpy as np

with open('randomForestPenguinClassifier.pkl','rb') as file:
    rf = pickle.load(file)
file.close()
with open('output.pkl','rb') as file:
    outputs_label = pickle.load(file)
file.close()

st.title('Penguin Classifier')
st.write('This app uses 6 inputs to predict the species of the penguin using the Palmers Penguin Dataset')


island = st.selectbox(label = 'Penguin Island', options = ['Biscoe','Dream','Torgerson'])
sex    = st.selectbox(label = 'Penguin Sex', options = ['male', 'female'])

bill_length = st.number_input(label = 'Enter the bill length', min_value = 30., max_value = 60., value = 45., step = 0.2)
bill_depth = st.number_input(label = 'Enter the bill depth', min_value = 10., max_value = 23., value = 15., step = 0.2)
flipper_length = st.number_input(label = 'Enter the flipper length',min_value = 170., max_value = 231., value = 200., step = 1.)
body_mass = st.number_input(label = 'Enter the body mass',min_value = 2800, max_value = 6300, value = 4000, step = 50)

# st.write(f"The user selected the following Penguin Island : {island} and Penguin Sex : {sex}")
# st.write(f"The user inputs are \n Bill Length = {bill_length} \n Bill Depth : {bill_depth} \n Flipper Length : {flipper_length} \n Body Mass : {body_mass}")

penguin_gender = 1 if sex == 'male' else 0

penguin_island = 0
if island == 'Biscoe':
    penguin_island = 0
elif island == 'Dream':
    penguin_island = 1;
else:
    penguin_island = 2



if st.button('re-run'):
    xdata = np.array([penguin_island,bill_length, bill_depth, flipper_length, body_mass, penguin_gender]).reshape(1, -1) 
    print(xdata)
    ypred = rf.predict(xdata)
    st.write(ypred)
    st.subheader(f'The penguin species is {outputs_label[np.argmax(ypred)]} based on the user input')

