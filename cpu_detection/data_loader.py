import cx_Oracle
import random
import math
import os
from dotenv import load_dotenv

load_dotenv()

username = os.getenv('DB_USERNAME')
password = os.getenv('DB_PASSWORD')
host = os.getenv('DB_HOST')
port = int(os.getenv('DB_PORT'))
service_name = os.getenv('DB_SERVICE_NAME')

dsn = cx_Oracle.makedsn(
    host=host,
    port=port,
    service_name=service_name
)

def insert_batch(cursor, sql, data_list):
    cursor.executemany(sql, data_list)

def get_time_features(day_minute):
    time_ratio = day_minute / 1440
    angle = 2 * math.pi * time_ratio
    return round(math.sin(angle), 4), round(math.cos(angle), 4)

def generate_clean_sample(minute):
    time_sin, time_cos = get_time_features(minute)
    request_rate = int(100 + 900 * abs(math.sin(2 * math.pi * minute / 1440)))  # request quantity in a minute
    http_errors = int(random.gauss(1, 0.5))  # http request with errors
    http_errors = max(http_errors, 0)
    cpu_usage = round(0.05 * request_rate + 1.5 * http_errors + random.uniform(-2, 2), 2)
    return {
        'cpu_usage': cpu_usage,
        'request_rate': request_rate,
        'http_errors': http_errors,
        'time_sin': time_sin,
        'time_cos': time_cos
    }

def generate_anomalous_sample(minute):
    data = generate_clean_sample(minute)
    anomaly_type = random.choice(["spike", "low_cpu", "high_errors", "time_mismatch"])

    if anomaly_type == "spike":  # sudden and unexpected spike in CPU usage
        data['cpu_usage'] += random.uniform(30, 50)
    elif anomaly_type == "low_cpu":
        data['cpu_usage'] = max(0, data['cpu_usage'] - random.uniform(20, 30))
    elif anomaly_type == "high_errors":
        data['http_errors'] += random.randint(20, 50)
        data['cpu_usage'] += random.uniform(10, 20)
    elif anomaly_type == "time_mismatch":  # small traffic at peak hours or high traffic at off-peak hours
        data['request_rate'] = random.randint(20, 50)
        data['cpu_usage'] += random.uniform(10, 30)

    data['cpu_usage'] = round(data['cpu_usage'], 2)
    return data

def train_data(batch_per_hour=200):
    connection = cx_Oracle.connect(user=username, password=password, dsn=dsn)
    cursor = connection.cursor()

    insert_sql = """
    INSERT INTO train_data (cpu_usage, request_rate, http_errors, time_sin, time_cos)
    VALUES (:cpu_usage, :request_rate, :http_errors, :time_sin, :time_cos)
    """

    for hour in range(24):
        batch = []
        for _ in range(batch_per_hour):
            minute = random.randint(hour * 60, (hour + 1) * 60 - 1)
            data = generate_clean_sample(minute)
            batch.append(data)
        insert_batch(cursor, insert_sql, batch)

    connection.commit()
    cursor.close()
    connection.close()
    print("Training data generated.")

def test_data(batch_per_hour=50, anomaly_ratio=0.2):
    connection = cx_Oracle.connect(user=username, password=password, dsn=dsn)
    cursor = connection.cursor()

    insert_sql = """
    INSERT INTO test_data (cpu_usage, request_rate, http_errors, time_sin, time_cos, is_anomaly)
    VALUES (:cpu_usage, :request_rate, :http_errors, :time_sin, :time_cos, :is_anomaly)
    """

    for hour in range(24):
        batch = []
        for _ in range(batch_per_hour):
            minute = random.randint(hour * 60, (hour + 1) * 60 - 1)
            if random.random() < anomaly_ratio:
                data = generate_anomalous_sample(minute)
                data['is_anomaly'] = 1
            else:
                data = generate_clean_sample(minute)
                data['is_anomaly'] = 0
            batch.append(data)
        insert_batch(cursor, insert_sql, batch)

    connection.commit()
    cursor.close()
    connection.close()
    print("Test data generated.")
        
if __name__ == "__main__":
    train_data()
    test_data()
    
    