#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import pandas_datareader.data as data
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import streamlit as st


st.title("Stock Trend Prediction")
user_input = st.text_input("Enter Stock Ticker", "AAPL")
start = dt.datetime(2010, 1, 1)
end = dt.datetime.today()

df = data.DataReader(user_input, 'stooq', start, end)

df = df.iloc[::-1]
df.reset_index(inplace=True)

# Describing data
st.subheader("Stock Data from 2010 till now")
st.write(df.describe())

# Visualizing Closing price
st.subheader("Closing price vs Time")
fig = plt.figure(figsize=(12, 6))
sns.lineplot(data = df, y = df.Close, x = df.Date)
plt.ylabel("Closing Price")
plt.xlabel("Year")
st.pyplot(fig)

# Defining 100 and 200 days moving averages
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()

# Visualizing 100 and 200 day moving average
st.subheader("Closing price with 100 and 200 day moving average vs Time")
fig = plt.figure(figsize = (12, 6))
plt.plot(df.Close)
plt.plot(ma100, 'r', label = "100 day MA")
plt.plot(ma200, 'g', label = "200 day MA")
plt.legend()
st.pyplot(fig)

# Train-test split
df_train = pd.DataFrame(df.Close[0:int(len(df)*0.75)])
df_test = pd.DataFrame(df.Close[int(len(df)*0.75):int(len(df))])

# Scaling
scaler = MinMaxScaler(feature_range=(0, 1))
df_train_array = scaler.fit_transform(df_train)
df_test_array = scaler.fit_transform(df_test)

# X and Y Train
x_train = []
y_train = []

for i in range(100, df_train_array.shape[0]):
    x_train.append(df_train_array[i-100: i])
    y_train.append(df_train_array[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# Loading model
model = load_model("stock_trend_predictor")

# 
past_100_days = df_train.tail(100)
final_df = past_100_days.append(df_test, ignore_index=True)

input_data = scaler.fit_transform(final_df)

# X and Y test
x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])
    
x_test, y_test = np.array(x_test), np.array(y_test)

# Predicting
y_pred = model.predict(x_test)

# Rescaling
scale_factor = 1/0.0074669
y_test = y_test*scale_factor
y_pred = y_pred*scale_factor

st.subheader("Original vs Predicted Price")
fig = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_pred, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)