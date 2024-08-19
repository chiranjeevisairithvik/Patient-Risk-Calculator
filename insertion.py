import sqlite3

# Connect to the database
conn = sqlite3.connect('database.db')

# Create a cursor object
cursor = conn.cursor()

# Insert data into the Users table
cursor.execute('''
INSERT INTO Users (name, age, email, mobile_number, password)
VALUES ('Priyatham', 20, 'itsmepriyatham@gmail.com', '12345678', 'Binnu123')
''')


# Commit the changes and close the connection
conn.commit()
conn.close()

print("Data inserted successfully.")
