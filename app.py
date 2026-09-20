import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Attrition Predictor", page_icon="📊",layout="wide")


model = joblib.load('rf_model.pkl')
model_columns = joblib.load('model_columns.pkl')


st.title("📊 Employee Attrition Predictor")
st.markdown("This tool leverages a Random Forest classifier trained with SMOTE ""to predict employee flight risk and drive proactive retention strategies.")
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
      st.subheader("👤 Employee Details")
      age = st.number_input("Age", min_value=18,max_value=65)
      department = st.selectbox("Department",["Sales","Research & Development", "Human Resources"])
      distance = st.number_input("Distance from Home (km)", min_value=1,max_value=50)

with col2:
     st.subheader("📈 Career & Tenure Metrics")
     total_years = st.number_input("Total Working Years", min_value=0, max_value=40)
     years_at_company = st.number_input("Years at Company",min_value=0, max_value=40)

with col3:
	st.subheader("💼Job & Compensation")
	monthly_income = st.number_input("Monthly Income ($)", min_value=1000, max_value=25000)
	job_satisfaction = st.selectbox("Job Satisfaction Rating", [1, 2, 3, 4], index=3)
	over_time = st.selectbox("Works Overtime?", ["Yes","No"], index=1)


st.markdown("---")

if st.button("🚀 Predict Flight Risk", type="primary", use_container_width=True):
	input_data = pd.DataFrame({'Age': [age],'Department': [department],'DistanceFromHome': [distance],'TotalWorkingYears': [total_years],'YearsAtCompany': [years_at_company],'MonthlyIncome': [monthly_income],'JobSatisfaction': [job_satisfaction],'Overtime': [over_time] })

	input_data['Loyalty_Ratio'] = input_data['YearsAtCompany'] / input_data['TotalWorkingYears']
	input_data['Loyalty_Ratio'] = input_data['Loyalty_Ratio'].fillna(0)

	input_data['EnvironmentSatisfaction'] = 4
	input_data['JobInvolvement'] = 4
	input_data['WorkLifeBalance'] = 4
	input_data['RelationshipSatisfaction'] = 4
	input_data['PerformanceRating'] = 4
	input_data['StockOptionLevel'] = 3
	input_data['NumCompaniesWorked'] = 1
	input_data['TrainingTimesLastYear'] = 3
	input_data['BusinessTravel'] = 'Non-Travel'
	input_data['MaritalStatus'] = 'Married'
	input_data['JobRole'] = 'Manager'
	input_data['JobLevel'] = 5

	input_encoded = pd.get_dummies(input_data)
	input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

	prediction = model.predict(input_encoded)
	probabilities = model.predict_proba(input_encoded)[0]

	st.subheader("📋 Evaluation Results")
	res_col1, res_col2 = st.columns([2, 1])

	with res_col1:
		if prediction[0] == 1:
			st.error("⚠️ **High Risk:** This employee is likely to resign.")
		else:
			st.success("✅ **Low Risk:** This employee is likely to stay.")
	with res_col2:
		if prediction[0] == 1:
			st.metric(label="AI Resignation Probability", value=f"{probabilities[1] * 100:.1f}%")
		else:
			st.metric(label="AI Retention Probability", value=f"{probabilities[0] * 100:.1f}%")