import sqlite3
import pandas as pd
import hashlib

def hash_password(password):
    """Creates a secure SHA-256 hash for passwords"""
    return hashlib.sha256(password.encode()).hexdigest()

def initialize_database():
    print("⚙️ Initializing Enterprise Relational Database...")
    
    # 1. Connect to SQLite (This creates the file 'enterprise_backend.db')
    conn = sqlite3.connect('enterprise_backend.db')
    cursor = conn.cursor()

    # 2. Create a secure Users table for dynamic authentication
    print("🔐 Building secure User Authentication table...")
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL
    )
    ''')

    # 3. Create the automated alerts log table
    print("📩 Building Automation Alerts table...")
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS system_alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        alert_type TEXT NOT NULL,
        message TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    # 4. Insert the default Admin user (with a hashed password!)
    admin_hash = hash_password("iub2026")
    try:
        cursor.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)", 
                       ('admin', admin_hash, 'System Administrator'))
        print("✅ Default Admin securely provisioned.")
    except sqlite3.IntegrityError:
        print("⚠️ Admin user already exists in database.")

    # 5. Migrate the CSV data into a real SQL table
    print("📦 Migrating static CSV data into live SQL table...")
    try:
        df = pd.read_csv('FYP_Perfect_Retail_Data.csv')
        df.to_sql('ecommerce_sales', conn, if_exists='replace', index=False)
        print("✅ Data migration complete. App is now database-driven.")
    except FileNotFoundError:
        print("⚠️ CSV file not found. Ensure 'FYP_Perfect_Retail_Data.csv' is in the directory.")

    conn.commit()
    conn.close()
    print("🚀 Backend Architecture Initialization Complete!")

if __name__ == "__main__":
    initialize_database()
