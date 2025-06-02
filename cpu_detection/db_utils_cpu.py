import cx_Oracle
import os
import numpy as np
from dotenv import load_dotenv

load_dotenv()

username = os.getenv('DB_USERNAME')
password = os.getenv('DB_PASSWORD')
host = os.getenv('DB_HOST')
port = int(os.getenv('DB_PORT', 1521))
service_name = os.getenv('DB_SERVICE_NAME')


dsn = cx_Oracle.makedsn(
    host=host,
    port=port,
    service_name=service_name
)


def fetch_train_data(): # fetching training data from the database from the train_data table
    connection = cx_Oracle.connect(user=username, password=password, dsn=dsn)
    if not connection:
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    cursor = connection.cursor()
    try:
        cursor.execute("SELECT cpu, users, time_sin, time_cos FROM train_data ORDER BY id") 
        rows = cursor.fetchall()
        if not rows:
            return np.array([]), np.array([]), np.array([]), np.array([])

        # saving data to numpy arrays
        cpu_data = np.array([row[0] for row in rows], dtype=np.float32)
        users_data = np.array([row[1] for row in rows], dtype=np.float32)
        time_sin_data = np.array([row[2] for row in rows], dtype=np.float32)
        time_cos_data = np.array([row[3] for row in rows], dtype=np.float32)
        
        return cpu_data, users_data, time_sin_data, time_cos_data
    except cx_Oracle.DatabaseError as e:
        print(f"Błąd podczas pobierania danych treningowych: {e}")
        return np.array([]), np.array([]), np.array([]), np.array([])
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

# same action as above but for test data
def fetch_test_data():
    connection = cx_Oracle.connect(user=username, password=password, dsn=dsn)

    cursor = connection.cursor()
    try:
        cursor.execute("SELECT cpu, users, time_sin, time_cos, is_anomaly FROM test_data ORDER BY id")
        rows = cursor.fetchall()
        if not rows:
            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

        cpu_data = np.array([row[0] for row in rows], dtype=np.float32)
        users_data = np.array([row[1] for row in rows], dtype=np.float32)
        time_sin_data = np.array([row[2] for row in rows], dtype=np.float32)
        time_cos_data = np.array([row[3] for row in rows], dtype=np.float32)
        is_anomaly_db = np.array([row[4] for row in rows], dtype=np.int32)
        
        return cpu_data, users_data, time_sin_data, time_cos_data, is_anomaly_db
    except cx_Oracle.DatabaseError as e:
        print(f"Błąd podczas pobierania danych testowych: {e}")
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

if __name__ == '__main__':
    # cpu_data, users_data, time_sin_data, time_cos_data
    train_cpu, train_users, train_time_sin, train_time_cos = fetch_train_data()
    if train_cpu.size > 0:
        print(f"Downloaded {len(train_cpu)} train records.")
    else:
        print("An error occurred while fetching training data.")

    test_cpu, test_users, test_time_sin, test_time_cos = fetch_test_data()
    if test_cpu.size > 0:
        print(f"Downloaded {len(test_cpu)} test records.")
    else:
        print("An error occurred while fetching test data.")