
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model and preprocessor (if any)
model = joblib.load('logistic_regression_model.joblib')

# Define the expected feature columns from training (excluding 'ProdTaken')
# This list must match the columns after one-hot encoding during training
expected_columns = [
    'Age', 'CityTier', 'DurationOfPitch', 'NumberOfPersonVisiting',
    'NumberOfFollowups', 'PreferredPropertyStar', 'NumberOfTrips',
    'Passport', 'PitchSatisfactionScore', 'OwnCar',
    'NumberOfChildrenVisiting', 'MonthlyIncome',
    'TypeofContact_Self Enquiry', 'Occupation_Housewife',
    'Occupation_Salaried', 'Occupation_Small Business', 'Gender_Male',
    'ProductPitched_Deluxe', 'ProductPitched_King', 'ProductPitched_Standard',
    'ProductPitched_Super Deluxe', 'MaritalStatus_Married', 'MaritalStatus_Single',
    'MaritalStatus_Unmarried', 'Designation_Executive', 'Designation_Manager',
    'Designation_Senior Manager', 'Designation_VP'
]

# Streamlit App Title
st.title('Wellness Tourism Package Purchase Predictor')

st.write("Enter customer details to predict if they will purchase the Wellness Tourism Package.")

# Create input fields for each feature
# Numerical inputs
age = st.slider('Age', 18, 80, 30)
city_tier = st.selectbox('City Tier', [1, 2, 3])
duration_of_pitch = st.slider('Duration of Pitch (minutes)', 1, 30, 10)
num_person_visiting = st.slider('Number of Persons Visiting', 1, 10, 2)
num_followups = st.slider('NumberOfFollowups', 0, 10, 3)
preferred_property_star = st.slider('Preferred Property Star Rating', 1, 5, 3)
num_trips = st.slider('NumberOfTrips', 1, 10, 2)
passport = st.selectbox('Has Passport?', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
pitch_satisfaction_score = st.slider('Pitch Satisfaction Score', 1, 5, 3)
own_car = st.selectbox('Owns Car?', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
num_children_visiting = st.slider('NumberOfChildrenVisiting', 0, 5, 0)
monthly_income = st.number_input('Monthly Income', min_value=0.0, value=25000.0)

# Categorical inputs (will be one-hot encoded)
type_of_contact = st.selectbox('Type of Contact', ['Company Invited', 'Self Enquiry'])
occupation = st.selectbox('Occupation', ['Salaried', 'Small Business', 'Freelancer', 'Housewife'])
gender = st.selectbox('Gender', ['Male', 'Female'])
product_pitched = st.selectbox('Product Pitched', ['Basic', 'Deluxe', 'King', 'Standard', 'Super Deluxe'])
marital_status = st.selectbox('Marital Status', ['Single', 'Married', 'Divorced', 'Unmarried'])
designation = st.selectbox('Designation', ['Executive', 'Manager', 'Senior Manager', 'AVP', 'VP', 'Director'])

# Create a dictionary from inputs
input_data = {
    'Age': age,
    'CityTier': city_tier,
    'DurationOfPitch': duration_of_pitch,
    'NumberOfPersonVisiting': num_person_visiting,
    'NumberOfFollowups': num_followups,
    'PreferredPropertyStar': preferred_property_star,
    'NumberOfTrips': num_trips,
    'Passport': passport,
    'PitchSatisfactionScore': pitch_satisfaction_score,
    'OwnCar': own_car,
    'NumberOfChildrenVisiting': num_children_visiting,
    'MonthlyIncome': monthly_income,
    'TypeofContact_Self Enquiry': 1 if type_of_contact == 'Self Enquiry' else 0,
    'Occupation_Housewife': 1 if occupation == 'Housewife' else 0,
    'Occupation_Salaried': 1 if occupation == 'Salaried' else 0,
    'Occupation_Small Business': 1 if occupation == 'Small Business' else 0,
    'Gender_Male': 1 if gender == 'Male' else 0,
    'ProductPitched_Deluxe': 1 if product_pitched == 'Deluxe' else 0,
    'ProductPitched_King': 1 if product_pitched == 'King' else 0,
    'ProductPitched_Standard': 1 if product_pitched == 'Standard' else 0,
    'ProductPitched_Super Deluxe': 1 if product_pitched == 'Super Deluxe' else 0,
    'MaritalStatus_Married': 1 if marital_status == 'Married' else 0,
    'MaritalStatus_Single': 1 if marital_status == 'Single' else 0,
    'MaritalStatus_Unmarried': 1 if marital_status == 'Unmarried' else 0,
    'Designation_Executive': 1 if designation == 'Executive' else 0,
    'Designation_Manager': 1 if designation == 'Manager' else 0,
    'Designation_Senior Manager': 1 if designation == 'Senior Manager' else 0,
    'Designation_VP': 1 if designation == 'VP' else 0
}

# Create a DataFrame from the input data
input_df = pd.DataFrame([input_data])

# Ensure all expected columns are present, fill missing with 0 (for one-hot encoded columns not selected)
for col in expected_columns:
    if col not in input_df.columns:
        input_df[col] = 0

# Reorder columns to match the training data's feature order
input_df = input_df[expected_columns]

# Predict button
if st.button('Predict Purchase'):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    st.write(f"### Prediction Result:")
    if prediction[0] == 1:
        st.success("The customer is likely to purchase the Wellness Tourism Package.")
    else:
        st.info("The customer is unlikely to purchase the Wellness Tourism Package.")

    st.write(f"**Probability of Purchase (Yes):** {prediction_proba[0][1]:.2f}")
    st.write(f"**Probability of Not Purchasing (No):** {prediction_proba[0][0]:.2f}")
