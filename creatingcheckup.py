import sqlite3

def create_checkup_table():
    conn = sqlite3.connect('database.db')  # Connect to the SQLite database file
    cursor = conn.cursor()

    # Create the checkup table with appropriate columns
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS checkup_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            age INTEGER,
            gender INTEGER,
            rbc_count REAL,
            wbc_count REAL,
            systole INTEGER,
            diastole INTEGER,
            heart_rate INTEGER,
            allergies INTEGER,
            blood_sugar REAL,
            bmi REAL,
            cholesterol REAL,
            steps_per_day INTEGER,
            calorie_intake INTEGER,
            smoking INTEGER,
            alcohol INTEGER,
            sleep_pattern INTEGER,
            claims VARCHAR(1000)
        )
    ''')

    conn.commit()
    conn.close()

if __name__ == "__main__":
    create_checkup_table()
    print("Database and checkup_data table created successfully.")
