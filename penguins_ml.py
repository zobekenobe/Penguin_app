import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

import pickle

# read in the data
df = pd.read_csv('Z:/4. LearningDT/Streamlit/Getting-Started-with-Streamlit-for-Data-Science-main/penguin_app/penguins.csv')

df.dropna(inplace = True)
# extracting the features and target
features = df.drop('species', axis = 1)
features.drop(columns = 'year', inplace = True)
outputs  = df['species']
le = LabelEncoder().fit(features['island'])
features['island'] = le.transform(features['island'])
features['sex'] = pd.get_dummies(features['sex'], drop_first = True)

outputs, specie_label = pd.factorize(outputs)

xtrain, xtest, ytrain, ytest = train_test_split(features,outputs,test_size = 0.2, random_state = 42)

# predicting model
rf = RandomForestClassifier()
rf.fit(xtrain, ytrain)

# save the model with Pickle
with open('model.pkl', 'wb') as file:
    pickle.dump({'model' : rf, 'island_encoder' : le, 'specie_label' : specie_label}, file)

file.close()

