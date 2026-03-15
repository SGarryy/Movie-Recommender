import os
import pyodbc
from dotenv import load_dotenv

load_dotenv()

def get_connection():
    server = os.getenv("DB_SERVER")
    database = os.getenv("DB_NAME")
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD")

    connection_string = (
        f"DRIVER={{ODBC Driver 17 for SQL Server}};"
        f"SERVER={server};"
        f"DATABASE={database};"
        f"UID={user};"
        f"PWD={password};"
        f"TrustServerCertificate=yes;"
    )

    try:
        conn = pyodbc.connect(connection_string)
        return conn
    except pyodbc.Error as e:
        raise ConnectionError(f"Database connection failed: {e}")