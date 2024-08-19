import sqlite3
import pandas as pd

# Correct the file path
file_path = r'C:\health-html\CLAIM_DATA_FINAL.csv'  # Using a raw string to avoid escape issues

# Load the CSV file
claim_data = pd.read_csv(file_path)

# Convert date_of_claim to date format and cost_of_service to float
claim_data['date_of_claim'] = pd.to_datetime(claim_data['date_of_claim'], format='%d-%m-%Y').dt.date
claim_data['cost_of_service'] = claim_data['cost_of_service'].astype(float)

# Correct the database path
database_path = r'C:\health-html\database.db'  # Adjust this to where you want the database to be stored

# Connect to the SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect(database_path)
cursor = conn.cursor()

# Create the claim_data table with appropriate data types
cursor.execute('''
CREATE TABLE IF NOT EXISTS claim_data (
    claim_id INTEGER PRIMARY KEY,
    patient_id INTEGER,
    date_of_claim DATE,
    service_type TEXT,
    diagnosis_code TEXT,
    procedure_code TEXT,
    cost_of_service REAL
)
''')

# Insert the data into the table
claim_data.to_sql('claim_data', conn, if_exists='replace', index=False, dtype={
    'claim_id': 'INTEGER',
    'patient_id': 'INTEGER',
    'date_of_claim': 'DATE',
    'service_type': 'TEXT',
    'diagnosis_code': 'TEXT',
    'procedure_code': 'TEXT',
    'cost_of_service': 'REAL'
})

# Commit the changes and close the connection
conn.commit()
conn.close()
