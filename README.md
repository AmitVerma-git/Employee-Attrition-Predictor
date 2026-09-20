# Enterprise Employee Attrition Predictor 🏢📊

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.24+-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-F7931E.svg)](https://scikit-learn.org/)

## Overview
This project is an end-to-end Machine Learning web application designed to transition Human Resources from reactive exit interviews to predictive retention modeling. Developed as a capstone project for the IBM Project-Based Experiential Learning Virtual Internship, the model analyzes employee demographic, tenure, and compensation data to predict flight risk with quantified AI confidence scores.

**👉 [Live Web Application](https://employee--attrition--predictor005.streamlit.app)**

## Technical Architecture
* **Algorithm:** Random Forest Classifier
* **Data Balancing:** Synthetic Minority Over-sampling Technique (SMOTE) was utilized during model training to address class imbalance inherent in HR attrition datasets.
* **Feature Engineering:** Implemented custom retention metrics, including a calculated `Loyalty_Ratio` (YearsAtCompany / TotalWorkingYears). Feature imputation was engineered into the production pipeline to inject healthy baseline metrics for hidden variables, ensuring stable real-world predictions.
* **Deployment:** Streamlit Community Cloud

## Repository Structure
* `app.py`: Main Streamlit application script containing the frontend UI and inference logic.
* `requirements.txt`: Environment dependencies required for deployment.
* `rf_model.pkl`: Serialized Random Forest machine learning model.
* `model_columns.pkl`: Serialized feature mapping to ensure strict categorical alignment during One-Hot Encoding.

## Local Setup Instructions
To run this application locally on your machine:
1. Clone the repository: `git clone <your-repo-url>`
2. Navigate to the directory: `cd <your-repo-directory>`
3. Install the required Python libraries: `pip install -r requirements.txt`
4. Launch the application: `streamlit run app.py`

## Author
**Amit Verma**  
*B.Tech Computer Science and Engineering | Meerut Institute of Engineering and Technology (MIET)*  
*Targeting roles in Software Engineering & AI Development*
