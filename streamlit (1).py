# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as pdr
import datetime
from keras.models import load_model
import streamlit as st

st.title('STOCK MARKET FORECASTING : TATA MOTORS ')

start=st.date_input('Start Date',value=datetime.datetime(2012, 7, 6))

end=st.date_input('End Date',value=datetime.datetime(2022, 7, 6))

# Importing Data for 10 years
start=datetime.datetime(2012, 7, 6)
end=datetime.datetime(2022, 7, 6)

user_input = st.text_input('Enter Stock name','TATAMOTORS.NS')
df = pdr.get_data_yahoo(user_input,start,end)

#Describe data
st.subheader('Data From yfinance :- 2012 to 2022')
st.write(df.describe())

# Data Visualization
st.subheader('[Close] Column:  Price vs Time')
fig = plt.figure(figsize=(12,6))
import plotly.express as px
fig = px.line(df.Close)
st.plotly_chart(fig)


st.subheader('[Close] Column: Price vs Time with 100ma')
ma100=df.Close.rolling(100).mean()
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add traces: Superimpose Plots  
fig.add_trace(
    go.Scatter(x=df.index, y=df['Close'], name="Close"),
    secondary_y=False,
)

fig.add_trace(
    go.Scatter(x=ma100.index, y=ma100, name="ma100"),
    secondary_y=True,
)
st.plotly_chart(fig)



st.subheader('[Close] Price vs Time with 100ma and 200ma')
ma200=df.Close.rolling(200).mean()
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add traces
fig.add_trace(
    go.Scatter(x=df.index, y=df['Close'], name="Close"),
    secondary_y=False,
)

fig.add_trace(
    go.Scatter(x=ma100.index, y=ma100, name="ma100"),
    secondary_y=True,
)

fig.add_trace(
    go.Scatter(x=ma200.index, y=ma200, name="ma200"),
    secondary_y=True,
)
st.plotly_chart(fig)

#Split Data into train & test

train=pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
test=pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])
print(train.shape)
print(test.shape)

# Scaling the data
from sklearn.preprocessing import MinMaxScaler
scaler= MinMaxScaler(feature_range=(0,1))

train_array=scaler.fit_transform(train)


#Spliting data inta x_train and y_train
x_train =[]
y_train =[]

for i in range (100,train_array.shape[0]):
    x_train.append(train_array[i-100: i])
    y_train.append(train_array[i, 0])
    
x_train,y_train = np.array(x_train),np.array(y_train)

#Load the model
model = load_model('keras_model.sdk')


#test part
past_100_days = train.tail(100)
final_df = past_100_days.append(test,ignore_index=True)
input_data=scaler.fit_transform(final_df)

x_test =[]
y_test =[]

for i in range (100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])
    
x_test,y_test = np.array(x_test),np.array(y_test)

y_predicted = model.predict(x_test)

scalerk = scaler.scale_
scale_factor = 1/scalerk[0]
y_predicted = y_predicted * scale_factor
y_test=y_test * scale_factor


#Final graph
st.subheader('Graph of Prediction vs Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label = 'Org Price')
plt.plot(y_predicted,'r',label='Pre Price')
plt.xlabel('Time')
plt.ylabel('price')
plt.legend()
st.pyplot(fig2)

#Prediction for next 10 days
tatak = pdr.get_data_yahoo('TATAMOTORS.NS', 
                          start=datetime.datetime(2012, 7, 6), 
                          end=datetime.datetime(2022, 7, 6))
tatak=tatak.filter(['Close'])
inputs = tatak[len(tatak) - len(train)- 100:].values
inputs = inputs.reshape(-1,1)
inputs = scaler.transform(inputs)
X_test = []
for i in range(100, 110):
    X_test.append(inputs[i-100:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = model.predict(X_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

tataa= pdr.get_data_yahoo('TATAMOTORS.NS', 
                          start=datetime.datetime(2022, 7, 7), 
                          end=datetime.datetime(2022, 7, 20))
tataa=tataa.filter(['Close'])
tataa=scaler.transform(tataa)
tataa = np.array(tataa)
tataa= scaler.inverse_transform(tataa)

#Graph of 10 days :Original vs Prediction
st.subheader('Graph of Prediction vs Original for next 10 days')
fig3 = plt.figure(figsize=(12,6))
plt.plot(tataa, color = 'black', label = 'TATA Stock Price')
plt.plot(predicted_stock_price, color = 'green', label = 'Predicted TATA Stock Price')
plt.title('TATA Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('TATA Stock Price')
plt.legend()
st.pyplot(fig3)









