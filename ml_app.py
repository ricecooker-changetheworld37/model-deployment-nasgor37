import streamlit as st
import numpy as np
import joblib
import os

# Dictionaries for encoding
gen = {'Male': 1, 'Female': 2}
e_mar = {'Yes': 1, 'No': 2}
grad = {'Yes': 1, 'No': 2}
prof = {'Healthcare': 1, 'Engineer': 2, 'Lawyer': 3, 'Entertainment': 4, 'Artist': 5,
        'Executive': 6, 'Doctor': 7, 'Homemaker': 8, 'Marketing': 9, 'Others': 10}
s_score = {'Low': 1, 'Average': 2, 'High': 3}
v1 = {'Cat_4': 4, 'Cat_6': 6, 'Cat_7': 7, 'Cat_3': 3, 'Cat_1': 1, 'Cat_2': 2, 'Unknown': 8,
      'Cat_5': 5}

def get_value(val, my_dict):
    return my_dict.get(val)

def run_ml_app():
    st.subheader('Rice Cooker Machine Learning Model Rifa_Fiola_Awani ver.01')
    gender = st.selectbox('Gender', ['Male', 'Female'])
    ever_married = st.selectbox('Ever Married', ['Yes', 'No'])
    graduated = st.selectbox('Graduated', ['Yes', 'No'])
    profession = st.selectbox('Profession', ['Healthcare', 'Engineer', 'Lawyer', 'Entertainment', 'Artist',
                                             'Executive', 'Doctor', 'Homemaker', 'Marketing', 'Others'])
    spending_score = st.selectbox('Spending Score', ['Low', 'Average', 'High'])
    var_1 = st.selectbox('Var 1', ['Cat_4', 'Cat_6', 'Cat_7', 'Cat_3', 'Cat_1', 'Cat_2', 'Unknown',
                                   'Cat_5'])
    age = st.number_input('Age', 18, 89)
    work_experience = st.number_input('Work Experience', 0, 35)
    family_size = st.number_input('Family Size', 1, 9)

    with st.expander("Your Selected Options"):
        result = {
            'Gender': gender,
            'Ever Married': ever_married,
            'Graduated': graduated,
            'Profession': profession,
            'Spending Score': spending_score,
            'Var 1': var_1,
            'Age': age,
            'Work Experience': work_experience,
            'Family Size': family_size
        }
        st.write(result)

    # Encode categorical variables
    encoded_result = [
        get_value(gender, gen),
        get_value(ever_married, e_mar),
        get_value(graduated, grad),
        get_value(profession, prof),
        get_value(spending_score, s_score),
        get_value(var_1, v1),
        age,
        work_experience,
        family_size
    ]

    st.write(encoded_result)

    st.subheader("Prediction Result:")
    single_array = np.array(encoded_result).reshape(1, -1)

    model = joblib.load(open(os.path.join('model_svm.pkl'), 'rb'))
    prediction = model.predict(single_array)

    if prediction == 0:
        st.success("User ini masuknya Segmentasi A")
    elif prediction == 1:
        st.success("User ini masuknya Segmentasi B")
    elif prediction == 2:
        st.success("User ini masuknya Segmentasi C")
    else:
        st.success("User ini masuknya Segmentasi D")

if __name__ == '__main__':
    run_ml_app()
