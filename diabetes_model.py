import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


diabetes = pd.read_excel("diabetes_data.xlsx")
diabetes = diabetes.drop(columns=(['SEX', 'AGE']))

st.write("""
# Diabetes Prediction App

This app predicts the quantitative measure of **diabetes disease progression** one year after baseline!

Data obtained from the [Diabetes Data](https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html).
""")


st.sidebar.header('User Input Parameters')


def user_input_features():
    bmi = st.sidebar.slider('BMI', 18, 42, 26)
    bp = st.sidebar.slider('Blood Pressure', 62, 133, 94)
    s1 = st.sidebar.slider('Total Serum Cholesterol', 97, 301, 189)
    s2 = st.sidebar.slider('Low-density Lipoproteins', 41, 242, 115)
    s3 = st.sidebar.slider('High-density Lipoproteins', 22, 99, 49)
    s4 = st.sidebar.slider('Total Cholesterol/HDL', 2, 9, 4)
    s5 = st.sidebar.slider('Serum Triglycerides Level', 4, 61048, 41314)
    s6 = st.sidebar.slider('Blood Sugar Level', 25, 346, 152)
    data = {'BMI': bmi,
            'BP': bp,
            'S1': s1,
            'S2': s2,
            'S3': s3,
            'S4': s4,
            'S5': s5,
            'S6': s6
            }
    features = pd.DataFrame(data, index=[0])
    return features


def progression(x):
    if x <= 77 or x <= 100:
        return 'Low'
    elif x <= 152 or x <= 240:
        return 'Medium'
    elif x <= 346:
        return 'High'


diabetes['Progression'] = diabetes['Y'].apply(progression)
target_mapper = {'Low': 0, 'Medium': 1, 'High': 2}


def target_encode(val):
    return target_mapper[val]


diabetes['Progression'] = diabetes['Progression'].apply(target_encode)

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

st.write("""
'BMI': Calculate Your Body Mass Index, 
'BP': Blood Pressure, 
'S1': Total Serum Cholesterol, 
'S2': Low-density Lipoproteins, 
'S3': High-density Lipoproteins, 
'S4': Total Cholesterol/HDL, 
'S5': Serum Triglycerides Level, 
'S6': Blood Sugar Level
""")

# Separating X and y
X = diabetes.drop(columns=(['Y', 'Progression']))
Y = diabetes['Progression']


clf = RandomForestClassifier()
clf.fit(X, Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Progression labels and their corresponding index number')
st.write(target_mapper)


st.subheader('Prediction')
st.write(f"#### The value predicted: **{prediction[0]}**")

st.subheader('Prediction Probability')
st.write(prediction_proba)

st.subheader('Plot Feature Importance')
st.write('Importance of each feature for the model prediction')
st.write("""'BMI': 0, 
'BP': 1, 
'S1': 2, 
'S2': 3, 
'S3': 4, 
'S4': 5, 
'S5': 6, 
'S6': 7""")


importance = clf.feature_importances_

st.bar_chart(importance)
