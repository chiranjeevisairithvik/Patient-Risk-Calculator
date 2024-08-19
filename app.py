from flask import Flask, jsonify, render_template, request, redirect, url_for, flash, session
import sqlite3
import pickle
import numpy as np
import os
import pandas as pd
import plotly
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from xgboost import XGBClassifier
import io
import pdfkit
from flask import send_file
from model import calculate_risk_score, predict_disease_and_risk, predict_stress_level
import plotly.express as px
import plotly.graph_objects as go
import json



app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Necessary for flash messages


# Function to check user credentials in the database
def check_user(email, password):
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM Users WHERE email = ? AND password = ?", (email, password))
    user = cursor.fetchone()
    conn.close()
    return user

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        user = check_user(email, password)
        
        if user:
            session['user_id'] = user[0]  # Store user ID in session
            session['name'] = user[1]  # Store user name in session
            return redirect(url_for('home'))
        else:
            flash('Invalid credentials, please try again.', 'danger')
            return redirect(url_for('login'))

    return render_template('login.html')


@app.route('/home')
def home():
    if 'user_id' in session:
        user_name = session.get('name', 'User')  # Get the user name from session
        return render_template('home.html', name=user_name)
    else:
        return redirect(url_for('login'))


def insert_user(name, age, email, mobile_number, password):
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO Users (name, age, email, mobile_number, password) VALUES (?, ?, ?, ?, ?)",
                   (name, age, email, mobile_number, password))
    conn.commit()
    conn.close()

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        age = request.form['age']
        email = request.form['email']
        mobile_number = request.form['mobile_number']
        password = request.form['password']

        # Insert user into the database
        insert_user(name, age, email, mobile_number, password)
        
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/forgot_password')
def forgot_password():
    return render_template('forgot_password.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('login'))  

@app.route('/about')
def about():
    return render_template('about.html')

import sqlite3

# Connect to the database (or create it if it doesn't exist)
conn = sqlite3.connect('database.db')

# Create a cursor object
cursor = conn.cursor()

# Create a table for storing checkup data with report_id and user_id
cursor.execute('''
    CREATE TABLE IF NOT EXISTS CheckupData (
        report_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        age INTEGER,
        gender TEXT,
        rbc_count REAL,
        wbc_count REAL,
        systole INTEGER,
        diastole INTEGER,
        heart_rate INTEGER,
        allergies TEXT,
        blood_sugar REAL,
        bmi REAL,
        cholesterol REAL,
        steps_per_day INTEGER,
        calorie_intake INTEGER,
        smoking INTEGER,
        alcohol INTEGER,
        sleep_pattern INTEGER,
        num_claims INTEGER,
        procedure_codes TEXT,
        FOREIGN KEY (user_id) REFERENCES Users(id)
    )
''')

# Commit the changes and close the connection
conn.commit()
conn.close()





@app.route('/checkup', methods=['GET', 'POST'])
def checkup():
 
    return render_template('checkup.html')


feature_columns = [
    'age', 'gender', 'rbc_count', 'wbc_count', 'systole', 'diastole', 
    'heart_rate', 'allergies', 'blood_sugar', 'bmi', 'cholesterol', 
    'steps_per_day', 'calorie_intake', 'smoking', 'alcohol', 
    'sleep_pattern', 'num_claims', 'service_type'
]

xgb_model = joblib.load('xgb_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')
model = joblib.load('model.pkl')  # Load your disease prediction model
mlb = joblib.load('mlb.pkl')  # MultiLabelBinarizer for diseases


@app.route('/predict', methods=['POST'])
def predict():
    user_name=session.get('name', 'User')
    user_id = session.get('user_id')
    # Extract input data from the form
    steps_per_day = request.form.get('steps_per_day')
    calorie_intake = request.form.get('calorie_intake')
    smoking = request.form.get('smoking')
    alcohol = request.form.get('alcohol')
    sleep_pattern = request.form.get('sleep_pattern')

    age = request.form.get('age')
    gender = request.form.get('gender')
    rbc_count = request.form.get('rbc_count')
    wbc_count = request.form.get('wbc_count')
    systole = request.form.get('systole')
    diastole = request.form.get('diastole')
    heart_rate = request.form.get('heart_rate')
    allergies = request.form.get('allergies')
    blood_sugar = request.form.get('blood_sugar')
    bmi = request.form.get('bmi')
    cholesterol = request.form.get('cholesterol')
    num_claims = request.form.get('num_claims')
    procedure_codes = request.form.get('procedure_codes')


   

    # Convert necessary fields to appropriate data types
    steps_per_day = int(steps_per_day)
    calorie_intake = int(calorie_intake)
    smoking = int(smoking)
    alcohol = int(alcohol)
    sleep_pattern = int(sleep_pattern)
    age = int(age)
    gender = int(gender)
    rbc_count = float(rbc_count)
    wbc_count = float(wbc_count)
    systole = int(systole)
    diastole = int(diastole)
    heart_rate = int(heart_rate)
    allergies = int(allergies)
    blood_sugar = float(blood_sugar)
    bmi = float(bmi)
    cholesterol = float(cholesterol)
    num_claims = int(num_claims)

    # Predict stress level
    predicted_stress_level = predict_stress_level(
        steps_per_day,
        calorie_intake,
        smoking,
        alcohol,
        sleep_pattern
    )

    # Prepare patient data for disease prediction
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

    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO CheckupData (user_id, age, gender, rbc_count, wbc_count, systole, diastole, heart_rate, allergies, blood_sugar, bmi, cholesterol, steps_per_day, calorie_intake, smoking, alcohol, sleep_pattern, num_claims, procedure_codes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (user_id, age, gender, rbc_count, wbc_count, systole, diastole, heart_rate, allergies, blood_sugar, bmi, cholesterol, steps_per_day, calorie_intake, smoking, alcohol, sleep_pattern, num_claims, procedure_codes))
    
    conn.commit()
    conn.close()

    # Predict diseases and calculate risk score
    predicted_disease, risk_score = predict_disease_and_risk(model, mlb, patient_data)

    # Pass the predictions to the result.html page
    return render_template('result.html',name=user_name, stress_level=predicted_stress_level, diseases=predicted_disease, risk_score=risk_score)

def predict_stress_level(steps_per_day, calorie_intake, smoking, alcohol, sleep_pattern):
    input_data = pd.DataFrame({
        'steps_per_day': [steps_per_day],
        'calorie_intake': [calorie_intake],
        'smoking': [smoking],
        'alcohol': [alcohol],
        'sleep_pattern': [sleep_pattern]
    })

    # Make prediction using the pre-trained model
    prediction_encoded = xgb_model.predict(input_data)[0]
    prediction_label = label_encoder.inverse_transform([prediction_encoded])[0]

    return prediction_label

def predict_disease_and_risk(model, mlb, patient_data):
    patient_df = pd.DataFrame([patient_data])

    # Predict diseases using the disease model
    prediction = model.predict(patient_df)
    predicted_labels = mlb.inverse_transform(prediction)
    
    # Calculate risk score based on the predicted diseases and patient data
    risk_score = calculate_risk_score(predicted_labels[0], patient_data)
    
    return predicted_labels[0], risk_score

def calculate_risk_score(diseases, patient_data):
    risk_factor = 0
    disease_risk_mapping = {
        'Diabetes': 2,
        'Obesity': 1.5,
        'Hyperlipidemia': 1.5,
        'Anemia': 1.2,
        'Cardiovascular': 2.5,
        'Asthma': 1.3,
        'Liver Disease': 2,
        'Hypertension': 2,
        'Healthy': 0
    }

    procedure_code_risk_mapping = {
        'P01': 1,
        'P02': 2,
        'P03': 3,
        'P04': 4,
        'P05': 5
    }

    for disease in diseases:
        risk_factor += disease_risk_mapping.get(disease, 1)

    risk_factor += patient_data['age'] // 100
    risk_factor += patient_data['num_claims'] // 10
    risk_factor += procedure_code_risk_mapping.get(patient_data['procedure_codes'], 1)

    return risk_factor


@app.route('/my_reports')
def my_reports():
    user_id = session.get('user_id')

    # Connect to the database and retrieve the user's checkup data
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT report_id, age, gender, rbc_count, wbc_count, systole, diastole, heart_rate, allergies, blood_sugar, bmi, cholesterol, steps_per_day, calorie_intake, smoking, alcohol, sleep_pattern, num_claims, procedure_codes
        FROM CheckupData WHERE user_id = ?
    ''', (user_id,))
    checkup_data = cursor.fetchall()
    conn.close()

    return render_template('my_reports.html', checkup_data=checkup_data)


@app.route('/diet_plan')
def diet_plan():
    user_name=session.get('name', 'User')
    return render_template('diet_plan.html',name=user_name)


@app.route('/feedback')
def feedback():
    return render_template('feedback.html')

@app.route('/thank_you')
def thank_you():
    return render_template('thank_you.html')

@app.route('/download_pdf', methods=['GET'])
def download_pdf():
    user_name = session.get('name', 'User')
    stress_level = request.args.get('stress_level')
    diseases = request.args.getlist('diseases')
    risk_score = request.args.get('risk_score')

    # Render the result page to HTML string
    html = render_template('result.html', name=user_name, stress_level=stress_level, diseases=diseases, risk_score=risk_score)
    
    # Convert HTML to PDF
    pdf = pdfkit.from_string(html, False)

    # Create a response object to send the PDF as a downloadable file
    response = send_file(io.BytesIO(pdf), download_name="report.pdf", as_attachment=True, mimetype='application/pdf')
    return response



@app.route('/preventive_measures', methods=['GET', 'POST'])
def preventive_measures():
    if request.method == 'POST':
        # Get the form data
        risk_level = int(request.form.get('risk_level'))
        diseases = request.form.getlist('diseases')

        # Generate preventive measures based on risk level and diseases
        preventive_measures = generate_preventive_measures(risk_level, diseases)

        return render_template('preventive_measures_result.html', preventive_measures=preventive_measures)

    return render_template('preventive_measures.html')

def generate_preventive_measures(risk_level, diseases):
    measures = []

    # General recommendations based on risk level
    if risk_level >= 4:
        measures.append("Follow a regular exercise routine suitable to your health.")
        measures.append("Consult with a healthcare provider regularly.")
    elif risk_level == 3:
        measures.append("Engage in moderate physical activity.")
        measures.append("Maintain a balanced diet with fruits and vegetables.")
    elif risk_level <= 2:
        measures.append("Ensure a healthy lifestyle with balanced nutrition and regular activity.")

    # Disease-specific preventive measures
    if 'Diabetes' in diseases:
        measures.append("Monitor your blood glucose levels.")
        measures.append("Incorporate fiber-rich foods into your meals.")

    if 'Heart Disease' in diseases:
        measures.append("Avoid saturated fats and cholesterol.")
        measures.append("Limit alcohol consumption.")

    if 'Hypertension' in diseases:
        measures.append("Reduce sodium intake in your diet.")
        measures.append("Increase potassium intake through fruits and vegetables.")

    if 'Liver Disease' in diseases:
        measures.append("Avoid alcohol and substances harmful to the liver.")
        measures.append("Consult your doctor before taking any new medications.")

    if 'Asthma' in diseases:
        measures.append("Avoid allergens and pollutants.")
        measures.append("Maintain a healthy weight.")

    if 'Obesity' in diseases:
        measures.append("Follow a calorie-controlled diet.")
        measures.append("Incorporate regular physical activity.")

    if 'Anemia' in diseases:
        measures.append("Incorporate iron-rich foods like lean meats and spinach.")
        measures.append("Consume vitamin C-rich foods to enhance iron absorption.")

    # Additional combined disease management recommendations
    if 'Diabetes' in diseases and 'Heart Disease' in diseases:
        measures.append("Follow a heart-healthy, low-sugar diet.")

    if 'Hypertension' in diseases and 'Heart Disease' in diseases:
        measures.append("Follow a DASH diet and reduce stress.")

    if 'Diabetes' in diseases and 'Obesity' in diseases:
        measures.append("Focus on a weight management plan with low-impact exercises.")

    return measures


df = pd.read_csv('data/merged.csv')


@app.route('/analysis')
def analysis():
    # Split the 'diseases' column into individual diseases
    disease_counts = df['diseases'].str.get_dummies(sep=', ').sum().sort_values(ascending=False)
    
    # Create a pie chart for diseases
    fig_pie = px.pie(values=disease_counts.values, names=disease_counts.index, title="Distribution of Diseases")

    # Convert the plot to JSON format
    graphJSON_pie = json.dumps(fig_pie, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('analysis.html', graphJSON_pie=graphJSON_pie)

@app.route('/disease/<disease_name>')
def disease_detail(disease_name):
    # Filter the dataframe for the selected disease
    filtered_df = df[df['diseases'].str.contains(disease_name)]

    # Visualization 1: Gender distribution for the disease
    fig_gender = px.histogram(filtered_df, x='gender', nbins=2, title=f"Gender Distribution for {disease_name}",
                              labels={'gender': 'Gender', 'count': 'Count'}, 
                              category_orders={'gender': [0, 1]})
    fig_gender.update_xaxes(tickvals=[0, 1], ticktext=['Female', 'Male'])

    # Visualization 2: Age distribution for the disease
    fig_age = px.histogram(filtered_df, x='age', nbins=10, title=f"Age Distribution for {disease_name}")

    # Convert the plots to JSON format
    graphJSON_gender = json.dumps(fig_gender, cls=plotly.utils.PlotlyJSONEncoder)
    graphJSON_age = json.dumps(fig_age, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('disease_detail.html', disease_name=disease_name,
                           graphJSON_gender=graphJSON_gender,
                           graphJSON_age=graphJSON_age)

@app.route('/result')
def result():
    user_name=session.get('name', 'User')
    return render_template('result.html',name=user_name)


if __name__ == '__main__':
    app.run(debug=True)



