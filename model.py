import pandas as pd
import keras
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

# # Load the dataset
# file_path = 'smalldatasets\lifestylee.csv'
# data = pd.read_csv(file_path)

# # Encode the categorical target variable
# label_encoder = LabelEncoder()
# data['stress_level_encoded'] = label_encoder.fit_transform(data['stress_level'])

# # Drop the patient_id column as it is not needed for prediction
# data_clean = data.drop(['patient_id', 'stress_level'], axis=1)

# # Split the data into features (X) and target (y)
# X = data_clean.drop('stress_level_encoded', axis=1)
# y = data_clean['stress_level_encoded']

# # Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Initialize and train the XGBoost classifier
# xgb_model = XGBClassifier(
#     use_label_encoder=False,
#     eval_metric='mlogloss',
#     random_state=42,
#     n_estimators=100,
#     max_depth=3,
#     learning_rate=0.1
# )

# xgb_model.fit(X_train, y_train)

# Function to predict stress level based on input parameters
def predict_stress_level(steps_per_day, calorie_intake, smoking, alcohol, sleep_pattern):
    input_data = pd.DataFrame({
        'steps_per_day': [steps_per_day],
        'calorie_intake': [calorie_intake],
        'smoking': [smoking],
        'alcohol': [alcohol],
        'sleep_pattern': [sleep_pattern]
    })

    # Make prediction
    prediction_encoded = xgb_model.predict(input_data)[0]
    prediction_label = label_encoder.inverse_transform([prediction_encoded])[0]

    return prediction_label

# # Clinical and claims data processing
# clinical_data_path = 'smalldatasets\clinicall.csv'
# clinical_data = pd.read_csv(clinical_data_path)

# claim_data_path = 'smalldatasets\claimm.csv'
# claim_data = pd.read_csv(claim_data_path)
# aggregated_claim_data = claim_data.groupby('patient_id').agg(
#     num_claims=pd.NamedAgg(column='claim_id', aggfunc='nunique')
# ).reset_index()

# merged_data = clinical_data.merge(aggregated_claim_data, left_on='patient_id', right_on='patient_id', how='left')

# merged_data['gender'] = merged_data['gender'].map({'Male': 1, 'Female': 0})
# merged_data['diseases'] = merged_data['diseases'].astype(str)
# merged_data['disease_list'] = merged_data['diseases'].apply(lambda x: x.split(', '))
# mlb = MultiLabelBinarizer()
# y = mlb.fit_transform(merged_data['disease_list'])

# X = merged_data.drop(columns=['patient_id', 'diseases', 'disease_list'])
# numerical_features = X.columns

# numerical_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='median')),
#     ('scaler', StandardScaler())
# ])

# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', numerical_transformer, numerical_features)
#     ])
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# model = Pipeline(steps=[('preprocessor', preprocessor),
#                         ('classifier', MultiOutputClassifier(xgb.XGBClassifier(eval_metric='mlogloss', n_estimators=50)))])

# model.fit(X_train, y_train)

# scaler = StandardScaler()
# scaler.fit(X)

# y_pred = model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# report = classification_report(y_test, y_pred, target_names=mlb.classes_, zero_division=0)

# print(f'Model Accuracy: {accuracy}')
# print('Classification Report:')
# print(report)

def predict_disease_and_risk(model, mlb, patient_data):
    patient_df = pd.DataFrame([patient_data], columns=X.columns)
    prediction = model.predict(patient_df)
    predicted_labels = mlb.inverse_transform(prediction)
    risk_score = calculate_risk_score(predicted_labels[0], patient_data)
    return predicted_labels[0], risk_score

def calculate_risk_score(diseases, patient_data):
    risk_factor = 0
    disease_risk_mapping = {
        'Diabetes': 2,
        'Obesity': 1.5,
        'Hyperlipidemia': 1.5,
        'Anemia': 1.2,
        'Cardiovascular': 3,
        'Asthma': 1.3,
        'Liver Disease': 2,
        'Hypertension': 2,
        'Healthy': 0
    }

    procedure_code_risk_mapping = {
        'P01': 1,
        'P02': 1,
        'P03': 1,
        'P04': 3,
        'P05': 4
    }


    for procedure_code in patient_data['procedure_codes']:
        risk_factor += procedure_code_risk_mapping.get(procedure_code, 1)

    return risk_factor

def get_patient_data():
    age = int(input("Enter age: "))
    gender = int(input("Enter gender (1 for Male, 0 for Female): "))
    rbc_count = float(input("Enter RBC count: "))
    wbc_count = float(input("Enter WBC count: "))
    systole = int(input("Enter systolic blood pressure: "))
    diastole = int(input("Enter diastolic blood pressure: "))
    heart_rate = int(input("Enter heart rate: "))
    allergies = int(input("Enter allergies (0 for No, 1 for Yes): "))
    blood_sugar = float(input("Enter blood sugar level: "))
    bmi = float(input("Enter BMI: "))
    cholesterol = float(input("Enter cholesterol level: "))
    steps_per_day = int(input("Enter steps per day: "))
    calorie_intake = int(input("Enter calorie intake: "))
    smoking = int(input("Enter smoking status (1 for yes, 0 for no): "))
    alcohol = int(input("Enter alcohol consumption status (1 for yes, 0 for no): "))
    sleep_pattern = int(input("Enter sleep pattern (1 for good, 0 for poor): "))

    # Ask for the number of claims
    num_claims = int(input("Enter the number of claims: "))
    procedure_codes = []

    for i in range(num_claims):
        procedure_code = input(f"Enter procedure code for claim {i + 1} (P01, P02, P03, P04, P05): ").strip()
        procedure_codes.append(procedure_code)

    # Construct patient data dictionary
    patient_data = {
        'age': age,
        'gender': gender,
        'rbc_count': rbc_count,
        'wbc_count': wbc_count,
        'systole': systole,
        'diastole': diastole,
        'heart_rate': heart_rate,
        'allergies': allergies,
        'blood_sugar': blood_sugar,
        'bmi': bmi,
        'cholesterol': cholesterol,
        'smoking': smoking,
        'alcohol': alcohol,
        'steps_per_day': steps_per_day,
        'sleep_pattern': sleep_pattern,
        'calorie_intake': calorie_intake,
        'num_claims': num_claims,
        'procedure_codes': procedure_codes
    }

    return patient_data

# # Collect patient data and make predictions
# single_patient_data = get_patient_data()
# predicted_disease, risk_score = predict_disease_and_risk(model, mlb, single_patient_data)
# predicted_stress_level = predict_stress_level(single_patient_data['steps_per_day'],
#                                               single_patient_data['calorie_intake'],
#                                               single_patient_data['smoking'],
#                                               single_patient_data['alcohol'],
#                                               single_patient_data['sleep_pattern'])
# print(f"Predicted Stress Level: {predicted_stress_level}")

# print(f'Risk Score for the patient: {min(int(risk_score), 5)}')
# print(f'Predicted Diseases for the patient: {", ".join(predicted_disease)}')


import joblib

# # Save the model
# joblib.dump(model, 'model.pkl')
# joblib.dump(mlb, 'mlb.pkl')
# joblib.dump(label_encoder, 'label_encoder.pkl')
# joblib.dump(scaler, 'scaler.pkl')
# joblib.dump(xgb_model, 'xgb_model.pkl')
# joblib.dump(data, 'data.pkl')
# joblib.dump(data_clean, 'data_clean.pkl')
# joblib.dump(X_train, 'X_train.pkl')
# joblib.dump(X_test, 'X_test.pkl')
# joblib.dump(y_train, 'y_train.pkl')
# joblib.dump(y_test, 'y_test.pkl')
# joblib.dump(merged_data, 'merged_data.pkl')
# joblib.dump(claim_data, 'claim_data.pkl')
# joblib.dump(clinical_data, 'clinical_data.pkl')
# joblib.dump(claim_data_path, 'claim_data_path.pkl')
# joblib.dump(clinical_data_path, 'clinical_data_path.pkl')
# joblib.dump(file_path, 'file_path.pkl')


