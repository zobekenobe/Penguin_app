import numpy as np
import streamlit as st
import pickle
import warnings
warnings.filterwarnings('ignore')

# load the model
def load_model():
    with open('model.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

model = data['model']
label_encoder = data['island_encoder']
specie_label = data['specie_label']

def show_predict_page():

    # get the input from the user
    # Enter the gender and the island
    penguin_gender = st.selectbox(label = "Choose the gender", options = ['male', 'female'])
    penguin_island = st.selectbox(label = "Choose the island", options = ['Biscoe', 'Dream','Torgersen'])

    bill_length = st.slider(label = 'Bill Length', min_value = 30., max_value = 60., value = 45., step = 0.2)
    bill_depth  = st.slider(label = 'Bill Depth',  min_value = 10., max_value = 23., value = 15., step = 0.2)
    flipper_length = st.slider(label = 'Flipper Length', min_value = 170., max_value = 231., value = 200., step = 1.)
    body_mass = st.slider(label = 'Body Mass', min_value = 2800, max_value = 6300, value = 4000, step = 50)

    # Encode the data
    island = label_encoder.transform([penguin_island])
    gender = 1 if penguin_gender == 'male' else 0
    
    if st.button('calculate'):
        x = np.array([[island, bill_length, bill_depth, flipper_length, body_mass, gender]])
        y = model.predict(x) 
        print(specie_label)
        print(y)
        st.subheader(f"The penguin is a {specie_label.values[y[-1]]}")
        # st.subheader(f"{y}")

