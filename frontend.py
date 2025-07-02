import pandas as pd
import numpy as np
import joblib
from datetime import date
import streamlit as st
import shap
import matplotlib.pyplot as plt


pipeline = joblib.load('dilapidations_pipeline.pkl')
model = pipeline.named_steps['regressor']
preprocessor = pipeline.named_steps['preprocessor']

st.title('Dilapidation Settlement Predictor')

st.markdown('Enter lease and propert details to predict the expected settlement amount.')

location = st.selectbox('Location', ['London', 'Manchester', 'Leeds'])
property_type = st.selectbox('Property Type', ['Office', 'Retail'])
survey_score = st.slider('Survey Score (1 = Poor, 10 = Excellent)', 1.0, 10.0, 5.0, 0.1)
size_sqft = st.number_input('Size (sqft)', min_value=100, max_value=100000, value=1000)
claim_amount = st.number_input('Original Claim Amount (£)', min_value=0, value=50000)
negotiated = st.selectbox('Was this negotiated?', ['Yes', 'No'])
lease_years = st.slider('Lease Length (Years)', 1, 30, 5)
repairing_oblig = st.selectbox('Repairing Obligation', ['Partial', 'Internal Only'])
epc_rating = st.selectbox('EPC Rating', ['A', 'B', 'C', 'D', 'E'])
year_build = st.number_input('Year Built', min_value=1800, max_value=date.today().year, value=2000)
schedule_of_condition = st.selectbox('Schedule of Condition', ['Yes', 'No'])
defects_identified = st.number_input('Defects Identified', min_value=0, value=0)
legal_dispute = st.selectbox('Legal Dispute', ['Yes', 'No'])
mediation_used = st.selectbox('Mediation Used', ['Yes', 'No'])
tenant_type = st.selectbox('Tenant Type', ['Corporate', 'Public Sector', 'Retail'])
tenant_industry = st.selectbox('Tenant Industry', ['Finance', 'Healthcare', 'Manufacturing', 'Retail'])
tenant_solvency = st.selectbox('Tenant Solvency', ['Strong', 'Weak'])
time_to_settlement_days = st.number_input('Time to Settlement (Days)', min_value=0, value=30)



if st.button('Predict Settlement Amount'):
    input_data = pd.DataFrame([{
        'Location': location,
        'Property_Type': property_type,
        'Survey_Score': survey_score,
        'Size_sqft': size_sqft,
        'Original_Claim_Amount': claim_amount,
        'Negotiated': 1 if negotiated == 'Yes' else 0,
        'Lease_Length_Years': lease_years,
        'Repairing_Obligation': repairing_oblig,
        'EPC_Rating': epc_rating,
        'Year_Built': year_build,
        'Schedule_of_Condition': 1 if schedule_of_condition == 'Yes' else 0,
        'Defects_Identified': defects_identified,
        'Legal_Dispute': 1 if legal_dispute == 'Yes' else 0,
        'Mediation_Used': 1 if mediation_used == 'Yes' else 0,
        'Tenant_Type': tenant_type,
        'Tenant_Industry': tenant_industry,
        'Tenant_Solvency': tenant_solvency,
        'Time_to_Settlement_Days': time_to_settlement_days
    }])

    # Prediction
    prediction = pipeline.predict(input_data)[0]
    st.success(f'📊 Predicted Settlement Amount: £{prediction:,.2f}\n(+/- 3635.57)')


