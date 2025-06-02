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
metrics = [
    {
        "time_min": 0,
        "time_max": 240,
        "user_min": 1000,
        "user_max": 1500
    },
    {
        "time_min": 240,
        "time_max": 360,
        "user_min": 1000,
        "user_max": 2000
    },
    {
        "time_min": 360,
        "time_max": 480,
        "user_min": 3000,
        "user_max": 6000
    },
    {
        "time_min": 480,
        "time_max": 600,
        "user_min": 5000,
        "user_max": 9000
    },
    {
        "time_min": 600,
        "time_max": 720,
        "user_min": 7000,
        "user_max": 11000
    },
    {
        "time_min": 720,
        "time_max": 840,
        "user_min": 6000,
        "user_max": 10000
    },
    {
        "time_min": 840,
        "time_max": 960,
        "user_min": 8000,
        "user_max": 12000
    },
    {
        "time_min": 960,
        "time_max": 1080,
        "user_min": 6000,
        "user_max": 9000
    },
    {
        "time_min": 1080,
        "time_max": 1200,
        "user_min": 3000,
        "user_max": 7000
    },
    {
        "time_min": 1200,
        "time_max": 1320,
        "user_min": 2000,
        "user_max": 4000
    },
    {
        "time_min": 1320,
        "time_max": 1440,
        "user_min": 1000,
        "user_max": 3000
    }
]

def train_data(batch = 2000):
    try:
        connection = cx_Oracle.connect(user=username, password=password, dsn=dsn)
        cursor = connection.cursor()

        for i in range(11):
            for j in range(batch):
                    sql_insert = """
                    insert into train_data (cpu, users, time_sin, time_cos) values (:cpu, :users, :time_sin, :time_cos)
                    """
                    
                    time = round(random.uniform(metrics[i]['time_min'], metrics[i]['time_max']), 1) / 1440
                    angle = 2 * math.pi * time
                    time_sin = math.sin(angle)
                    time_cos = math.cos(angle)
                    
                    users = random.randint(metrics[i]['user_min'], metrics[i]['user_max'])
                    
                    cpu = 0.001 * users + 10 * time_sin + 5 * time_cos + random.uniform(-2, 2)
                    
                    data = {
                        'cpu': round(cpu, 2),
                        'users': users,
                        'time_sin': round(time_sin, 4),
                        'time_cos': round(time_cos, 4)
                    }
                    
                    cursor.execute(sql_insert, data)
        connection.commit()
                    
    except cx_Oracle.DatabaseError as e:
        print(e)
        
    finally:
        cursor.close()
        connection.close()
        print("Connection closed.")
        
        
        
def test_data(batch = 100):
    try:
        connection = cx_Oracle.connect(user=username, password=password, dsn=dsn)
        cursor = connection.cursor()

        for i in range(11):
            for j in range(batch):
                
                sql_insert = """
                insert into test_data (cpu, users, time_sin, time_cos, is_anomaly) values (:cpu, :users, :time_sin, :time_cos, :label)
                """
                
                time = round(random.uniform(metrics[i]['time_min'], metrics[i]['time_max']), 1) / 1440
                angle = 2 * math.pi * time
                time_sin = math.sin(angle)
                time_cos = math.cos(angle)
                
                users = random.randint(metrics[i]['user_min'], metrics[i]['user_max'])
                
                cpu_expected = 0.001 * users + 10 * time_sin + 5 * time_cos + random.uniform(-2, 2)
                
                if random.random() < 0.1:
                    multiplier = random.choice([1.15, 1.25, 1.3, 0.85, 0.8, 0.7])
                    cpu = cpu_expected * multiplier + random.uniform(-2, 2)
                    is_anomaly = 1
                else:
                    cpu = cpu_expected + random.uniform(-2, 2)
                    is_anomaly = 0
                    
                data = {
                    'cpu': round(cpu, 2),
                    'users': users,
                    'time_sin': round(time_sin, 4),
                    'time_cos': round(time_cos, 4),
                    'label': is_anomaly
                }
                
                cursor.execute(sql_insert, data)
        connection.commit()
            
    except cx_Oracle.DatabaseError as e:
        print(e)
        
    finally:
        cursor.close()
        connection.close()
        print("Connection closed.")
        
if __name__ == "__main__":
    train_data()
    test_data()