import cx_Oracle

username = 'osm_user'
password = 'BBBbOu123!#'
dsn = cx_Oracle.makedsn(

)

connection = cx_Oracle.connect(user=username, password=password, dsn=dsn)

try:
    cursor = connection.cursor()
    sql_insert = """
    insert into test_data (cpu, users, time_sin, time_cos) values (:cpu, :users, :time_sin, :time_cos)
    """
    data = {
        'cpu': 1.5,
        'users': 10,
        'time_sin': 0.1,
        'time_cos': 0.2
    }
    
    cursor.execute(sql_insert, data)
    connection.commit()
    print("Data inserted successfully.")
except cx_Oracle.DatabaseError as e:
    print(e)
finally:
    cursor.close()
    connection.close()
    print("Connection closed.")