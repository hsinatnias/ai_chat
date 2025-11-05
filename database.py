import sqlite3
from sqlite3 import Error

# Path to your SQLite database file
DB_PATH = "modules.db"

def create_connection():
    """Create a database connection to the SQLite database."""
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        return conn
    except Error as e:
        print(e)
    return conn

def create_table():
    """Create the modules table if it doesn't exist."""
    conn = create_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS modules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE
            );
        """)
        conn.commit()
    except Error as e:
        print(f"Error creating table: {e}")
    finally:
        if conn:
            conn.close()

def add_module(module_name: str):
    """Add a new module to the database."""
    conn = create_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO modules (name) VALUES (?)", (module_name,))
        conn.commit()
    except Error as e:
        print(f"Error adding module: {e}")
    finally:
        if conn:
            conn.close()

def get_modules():
    """Fetch all modules from the database."""
    conn = create_connection()
    modules = []
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM modules")
        rows = cursor.fetchall()
        modules = [row[0] for row in rows]
    except Error as e:
        print(f"Error fetching modules: {e}")
    finally:
        if conn:
            conn.close()
    return modules

def delete_module(module_name: str):
    """Delete a module from the database."""
    conn = create_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM modules WHERE name=?", (module_name,))
        conn.commit()
    except Error as e:
        print(f"Error deleting module: {e}")
    finally:
        if conn:
            conn.close()
