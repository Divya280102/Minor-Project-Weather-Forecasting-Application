import pandas as pd
import numpy as np
# import unicode
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pymysql as mq
mysql = mq.connect(host='localhost', user='root',
                   password='1234', database='newminiproject', port=3307)
mycursor = mysql.cursor()


def convertToBinaryData(filename):
    # Convert digital data to binary format
    with open(filename, 'rb') as file:
        binaryData = file.read()
    return binaryData

# states=["Telangana","Tripura","Uttar Pradesh","Uttarakhand","West Bengal"]
states=["Andaman and Nicobar Islan...","Andhra Pradesh","Arunachal Pradesh","Assam","Bihar","Chandigarh","Chhattisgarh","Dadra and Naga Haveli","Daman and Diu","Delhi","Gujarat","Haryana","Himachal Pradesh","Jammu and Kashmir","Jharkhand","Karnataka","Kerala","Ladakh","Lakshadweep","Madhya Pradesh","Maharashtra","Meghalaya","Mizoram","Nagaland","Odisha","Punducherry","Punjab","Rajasthan","Sikkim","Tamil Nadu","Telangana","Tripura","Uttar Pradesh","Uttarakhand","West Bengal"]

for i in states:
    i = i.upper()
    print(i)
    sql = 'SELECT *FROM weather_data WHERE State="%s";' % i
    mycursor.execute(sql)
    result = mycursor.fetchone()
    print(result)
    if (result != '()'):
        sql = "DELETE FROM weather_data WHERE State = '%s';" % i
        mycursor.execute(sql)
        mysql.commit()

    print("For state ", i)
    df = pd.read_csv(f"Mini project ML/{i} last30days.csv")
#     df = pd.read_csv("Haryana last30days.csv")
    print(df.head())
    df = df.replace('rain', 1)
    df["preciptype"].fillna("0", inplace=True)
    print(df.head())
    print(df['preciptype'])
    df = df[["name", "datetime", "precip", "tempmax", "tempmin", "temp", "windspeed", "dew", "humidity", "windgust", "preciptype",
             "visibility", "solarradiation", "solarenergy", "uvindex", "sunrise", "sunset", "conditions"]]
    df.head()
    len = df.shape[0]
    col_names = ['solarradiation', 'solarenergy', 'humidity', 'dew']
    df_temp_x = df.loc[:, col_names].values
    df_temp_y1 = df['tempmax'].values
    df_temp_y2 = df['tempmin'].values
    df_temp_y3 = df['temp'].values
    train_X, test_X, train_y, test_y = train_test_split(
        df_temp_x, df_temp_y1, test_size=0.2, random_state=4)
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(train_X, train_y)
    from sklearn.metrics import r2_score
    r2_a_score = model.score(test_X, test_y)
    print(r2_a_score*100, '%')
    r2_a_score = model.score(train_X, train_y)
    print(r2_a_score*100, '%')
    print("For state ", i)
    solar_rad = float(input("Enter today's solar radiation : "))
    solar_ene = float(input("Enter today's solar energy : "))
    humidity = float(input("Enter today's humidity : "))
    dew = float(input("Enter today's dew : "))
    Max_temp = model.predict([[solar_rad, solar_ene, humidity, dew]])[0]
    print("Maximum Temerature of today is :", Max_temp)
    train_X, test_X, train_y, test_y = train_test_split(
        df_temp_x, df_temp_y2, test_size=0.2, random_state=4)
    model.fit(train_X, train_y)
    r2_a_score = model.score(test_X, test_y)
    print(r2_a_score*100, '%')
    r2_a_score = model.score(train_X, train_y)
    print(r2_a_score*100, '%')
    Min_temp = model.predict([[solar_rad, solar_ene, humidity, dew]])[0]
    print("Minimum temperature of today is :", Min_temp)
    train_X, test_X, train_y, test_y = train_test_split(
        df_temp_x, df_temp_y3, test_size=0.2, random_state=4)
    model.fit(train_X, train_y)
    r2_a_score = model.score(test_X, test_y)
    print(r2_a_score*100, '%')
    r2_a_score = model.score(train_X, train_y)
    print(r2_a_score*100, '%')
    temp = model.predict([[solar_rad, solar_ene, humidity, dew]])[0]
    print("The commonly observe temperature is : ", temp)
    x = df.loc[len-4:, 'datetime'].values.tolist()
    print(x)
    print(type(x))
    print(type(x[0]))
    import datetime
    from datetime import date
    # Today = date.today()
    Today=datetime.datetime(2022,12,5)
    x.append(Today.strftime("%d-%m-%Y"))
    print(x)
    y = df.loc[len-4:, 'temp'].values.tolist()
    print(y)
    print(type(y))
    temp = float(temp)
    print("The temperature is : ", temp)
    y.append(temp)
    print(y)
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker='o', mec='r', mfc='r', linestyle='--', linewidth=2,
             markersize=6, label="Temp Line")
    plt.title("Temperature Plot With Time", fontsize=18)
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Temperature", fontsize=14)
    # plt.legend(loc=4)
    plt.savefig(f'images\{i}.png', bbox_inches='tight')
    plt.show()
    # photo = f'{i}.png'
    # print(photo)
    # print(f'{i}.png')
    # Picture1 = convertToBinaryData(f'{i}.png')
    # Picture1 = unicode(Picture1, "utf-8")
    # print(Picture1)

    # Wind
    col_names = ['windgust', 'dew']
    df_x = df.loc[:, col_names]
    df_y = df['windspeed']
    from sklearn.model_selection import train_test_split
    train_X, test_X, train_y, test_y = train_test_split(
        df_x, df_y, test_size=0.2, random_state=4)
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(train_X, train_y)
    r2_a_score = model.score(train_X, train_y)
    print("Train data score : ", r2_a_score*100, '%')
    r2_a_score = model.score(test_X, test_y)
    print("Test data score : ", r2_a_score*100, '%')

    # L1 regularization
    from sklearn import linear_model
    lesso_reg = linear_model.Lasso(alpha=50, max_iter=100, tol=0.1)
    lesso_reg.fit(train_X, train_y)
    lesso_score = lesso_reg.score(train_X, train_y)
    print("Train data score : ", lesso_score*100, '%')
    lesso_reg.fit(test_X, test_y)
    lesso_score = lesso_reg.score(test_X, test_y)
    print("Test data score : ", lesso_score*100, '%')

    # ridge regression
    from sklearn.linear_model import Ridge
    ridge_reg = Ridge(alpha=50, max_iter=100, tol=0.1)
    ridge_reg.fit(train_X, train_y)
    ridge_score = ridge_reg.score(train_X, train_y)
    print("Train data score : ", ridge_score*100, '%')
    ridge_reg.fit(test_X, test_y)
    ridge_score = ridge_reg.score(test_X, test_y)
    print("Test data score : ", ridge_score*100, '%')

    # print("Error below this")
    windgust = float(input('Enter the windgust : '))
    # print("Hello")
    wind_speed = ridge_reg.predict([[dew, windgust]])[0]
    print(wind_speed)

    # Condition
    df.replace(to_replace="Clear", value=1, inplace=True)
    df.replace(to_replace="Partially cloudy", value=2, inplace=True)
    df.replace(to_replace="Rain, Partially cloudy", value=3, inplace=True)
    df.replace(to_replace="Rain", value=4, inplace=True)
    df.replace(to_replace="Rain, Overcast", value=5, inplace=True)
    df.replace(to_replace="Overcast", value=5, inplace=True)
    print(df)
    col_names = ['temp', 'humidity', 'dew', 'windspeed',
                 'windgust', 'solarradiation', 'solarenergy']
    df_x = df.loc[:, col_names]
    df_y = df['conditions']
    train_X, test_X, train_y, test_y = train_test_split(
        df_x, df_y, test_size=0.2, random_state=4)
    model = LinearRegression()
    model.fit(train_X, train_y)
    print(train_X.shape)
    print(train_y.shape)
    from sklearn.metrics import r2_score
    r2_a_score = model.score(test_X, test_y)
    print("Test data score : ", r2_a_score*100, '%')
    r2_a_score = model.score(train_X, train_y)
    print("Train data score : ", r2_a_score*100, '%')
    temp = float(temp)
    humidity = float(humidity)
    dew = float(dew)
    wind_speed = float(wind_speed)
    windgust = float(windgust)
    solar_rad = float(solar_rad)
    solar_ene = float(solar_ene)
    cond = model.predict(
        [[temp, humidity, dew, wind_speed, windgust, solar_rad, solar_ene]])[0]
    # cond=cond[0]
    print(cond)
    cond = round(cond)
    print(cond)
    if (cond < 1):
        cond = 1
    if (cond > 5):
        cond = 5
    if (cond == 1):
        cond = "Clear"
    elif (cond == 2):
        cond = "Partially cloudy"
    elif (cond == 3):
        cond = "Rain, Partially cloudy"
    elif (cond == 4):
        cond = "Rain"
    else:
        cond = "Rain, Overcast"
    print(cond)

    df.replace(to_replace=1, value="Clear", inplace=True)
    df.replace(to_replace=2, value="Partially cloudy", inplace=True)
    df.replace(to_replace=3, value="Rain, Partially cloudy", inplace=True)
    df.replace(to_replace=4, value="Rain", inplace=True)
    df.replace(to_replace=5, value="Rain, Overcast", inplace=True)

    y = df.loc[len-4:, 'conditions'].values.tolist()
    print(y)
    print(type(y))
    print(cond)
    y.append(cond)
    print(y)
    plt.figure(figsize=(7, 2))
    plt.plot(x, y, marker='o', mec='r', mfc='r', linestyle='--', linewidth=2,
             markersize=6, label="Temp Line")
    plt.title("Condition Plot With Time", fontsize=18)
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Condition", fontsize=14)
    plt.savefig(f'images\{i}cond.png', bbox_inches='tight')
    plt.show()

    # try:
    # ins = "INSERT INTO weather_data (State,Today_Temp,TempVsDate) values(%s,%s,%s)"
    temp = "{:.2f}". format(temp)
    wind_speed = "{:.2f}". format(wind_speed)
    ins = "INSERT INTO weather_data (State,Today_Temp,Today_wind,Today_condition) values(%s,%s,%s,%s)"
    # t = (i, temp, Picture1)
    t = (i, temp, wind_speed, cond)
    print(t)
    mycursor.execute(ins, t)
    mysql.commit()
    print("insert data")
    # except:
    # print("Error...")
