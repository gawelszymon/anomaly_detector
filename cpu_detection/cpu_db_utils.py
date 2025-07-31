import os

import cx_Oracle
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


def fetch_train_data():

    # Fetch training data from the train_data table.
    # Returns:
    # cpu_usage, request_rate, http_errors, time_sin, time_cos (all np.arrays)

    connection = cx_Oracle.connect(user=username, password=password, dsn=dsn)
    if not connection:
        return tuple(np.array([]) for _ in range(5))
    cursor = connection.cursor()
    try:
        cursor.execute("SELECT cpu_usage, request_rate, http_errors, time_sin, time_cos FROM train_data")
        rows = cursor.fetchall()
        if not rows:
            return tuple(np.array([]) for _ in range(5))
        cpu_usage = np.array([row[0] for row in rows], dtype=np.float32)
        request_rate = np.array([row[1] for row in rows], dtype=np.float32)
        http_errors = np.array([row[2] for row in rows], dtype=np.float32)
        time_sin = np.array([row[3] for row in rows], dtype=np.float32)
        time_cos = np.array([row[4] for row in rows], dtype=np.float32)
        return cpu_usage, request_rate, http_errors, time_sin, time_cos
    except cx_Oracle.DatabaseError as e:
        print(f"Error while fetching training data: {e}")
        return tuple(np.array([]) for _ in range(5))
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

def fetch_test_data():
    """
    Fetch test data from the test_data table.
    Returns:
        cpu_usage, request_rate, http_errors, time_sin, time_cos, is_anomaly (all np.arrays)
    """
    connection = cx_Oracle.connect(user=username, password=password, dsn=dsn)
    cursor = connection.cursor()
    try:
        cursor.execute("SELECT cpu_usage, request_rate, http_errors, time_sin, time_cos, is_anomaly FROM test_data")
        rows = cursor.fetchall()
        if not rows:
            return tuple(np.array([]) for _ in range(6))
        cpu_usage = np.array([row[0] for row in rows], dtype=np.float32)
        request_rate = np.array([row[1] for row in rows], dtype=np.float32)
        http_errors = np.array([row[2] for row in rows], dtype=np.float32)
        time_sin = np.array([row[3] for row in rows], dtype=np.float32)
        time_cos = np.array([row[4] for row in rows], dtype=np.float32)
        is_anomaly = np.array([row[5] for row in rows], dtype=np.int32)
        return cpu_usage, request_rate, http_errors, time_sin, time_cos, is_anomaly
    except cx_Oracle.DatabaseError as e:
        print(f"Error while fetching test data: {e}")
        return tuple(np.array([]) for _ in range(6))
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

if __name__ == '__main__':
    train = fetch_train_data()
    if train[0].size > 0:
        print(f"Downloaded {len(train[0])} train records.")
    else:
        print("An error occurred while fetching training data.")

    test = fetch_test_data()
    if test[0].size > 0:
        print(f"Downloaded {len(test[0])} test records.")
    else:
        print("An error occurred while fetching test data.")
