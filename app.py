# import streamlit as st
# import numpy as np
# import tensorflow as tf
# from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
# import pandas as pd
# import pickle

# # loading the trained model
# model=tf.keras.models.load_model('model.h5')

# # load the encoders and scaler
# with open('label_encoder_gender.pkl', 'rb') as file:
#     label_encoder_gender=pickle.load(file)

# with open('label_encoder_geo.pkl', 'rb') as file:
#     label_encoder_geo=pickle.load(file)

# with open('scaler.pkl', 'rb') as file:
#     scaler=pickle.load(file)


# # streamlit app
# st.title('Customer churn prediction')

# #user input

# geography=st.selectbox('Geography', label_encoder_geo.categories_[0])
# #Uses .classes_ to show the original string labels (['Female', 'Male']) to the user
# gender=st.selectbox('Gender', label_encoder_gender.classes_)
# age=st.slider('Age', 18, 92)
# balance=st.number_input('Balance')
# credit_score=st.number_input('Credit Score')
# estimated_salary=st.number_input('Estimated Salary')
# tenure=st.slider('Tenure', 1, 10)
# num_of_products=st.slider('Number Of Products', 1, 4)
# has_cr_card=st.selectbox('Has Credit Card', [0, 1])
# is_active_member=st.selectbox('Is Active Member', [0,1])

# # prepare the input data
# input_data=pd.DataFrame({
#     'CreditScore':[credit_score],
#     'Gender':[label_encoder_gender.transform([gender])[0]],
#     'Age':[age], 
#     'Balance':[balance],
#     'Tenure':[tenure],
#     'NumOfProducts':[num_of_products],
#     'HasCrCard':[has_cr_card],
#     'IsActiveMember':[is_active_member],
#     'EstimatedSalary':[estimated_salary]
# })

# # one hot encode 'geography'

# # transform(...): Transforms 'France' to a one-hot encoded vector like [1, 0, 0].
# geo_encoded=label_encoder_geo.transform([[geography]]).toarray()

# # Wraps the encoded NumPy array in a DataFrame for easier inspection and merging.
# geo_encoded_df=pd.DataFrame(geo_encoded, columns=label_encoder_geo.get_feature_names_out(['Geography']))

# # combine onehot encoded columns with input data
# input_data=pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# # scale the input data
# input_data_scaled=scaler.transform(input_data)

# # predict churn
# prediction=model.predict(input_data_scaled)
# prediction_proba=prediction[0][0]

# if(prediction_proba>0.5):
#     st.write('customer is likely to churn')

# else:
#     st.write('customer is not likely to churn')







import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load the encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('label_encoder_geo.pkl', 'rb') as file:
    label_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Manually specify expected feature order used during training
expected_features = [
    'CreditScore', 'Gender', 'Age', 'Balance', 'Tenure',
    'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
    'Geography_France', 'Geography_Germany', 'Geography_Spain'
]

# Streamlit app
st.title('Customer Churn Prediction')

# User inputs
geography = st.selectbox('Geography', label_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 1, 10)
num_of_products = st.slider('Number Of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Balance': [balance],
    'Tenure': [tenure],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode Geography using get_dummies (simpler & avoids mismatch)
geo_dummies = pd.get_dummies([geography], prefix='Geography')
geo_df = pd.DataFrame(0, index=[0], columns=['Geography_France', 'Geography_Germany', 'Geography_Spain'])
geo_df.loc[0, geo_dummies.columns[0]] = 1

# Combine all features
input_data = pd.concat([input_data.reset_index(drop=True), geo_df], axis=1)

# Reorder columns to match training
input_data = input_data[expected_features]

# Scale
input_data_scaled = scaler.transform(input_data.values)

# Predict
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

# Display result

st.write(f'the churn probability is: {prediction_proba:.2f}')
if prediction_proba > 0.5:
    st.write('🔴 The customer is **likely to churn**.')
else:
    st.write('🟢 The customer is **not likely to churn**.')
