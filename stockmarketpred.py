import tkinter as tk
from tkinter import *
from tkinter import ttk
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense,Dropout,LSTM
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import time
import threading
from matplotlib.figure import Figure

plt.style.use('fivethirtyeight')
print("all modules imported")


win= tk.Tk()
win.title("stock market predictor")
win.geometry("1220x1000")

def submit():
  thread=threading.Thread(target=Training)
  thread.daemon = 1
  thread.start()  



def Training():
  global df
  STOCK_NAME=stocks.get()
  loading_data=Label(win,text=f"Loading data set of {STOCK_NAME}", font="Homoarakhn 10 bold")
  loading_data.grid(row=3,column=10)

  df=web.DataReader(STOCK_NAME,data_source='yahoo',start='2012-01-01',end='2020-05-01')

  Visualizing_label=Label(win,text="Visualizing your Dataset......", font="Homoarakhn 10 bold").grid(row=5,column=10)
  

  print("Visualizing Dataset")
  plt.figure(figsize=(16,10))
  plt.plot(df.index,df["Close"])
  plt.title("closing with date")
  plt.xlabel("Date")
  plt.ylabel("close")
  plt.show()
  


  thread=threading.Thread(target=visualize)
  thread.daemon = 1
  thread.start()  

  

  print("Preparing  dataset")
  data= df.filter(['Close']) #getting closing data from the dataframe
  dataset=data.values #covert the dataframe into array
  
  #getting no. of rows to convert into training data
  train_len= math.ceil(len(dataset)*.8)

  y1_test=dataset[train_len:,:]
  #training of dataset
  scaler= MinMaxScaler(feature_range=(0,1))
  scaled_data= scaler.fit_transform(dataset)
  len(scaled_data)

  #spliting of data into test and train
  train_data= scaled_data[:train_len, : ]
  testing_data=scaled_data[train_len-60:, :]

  def preparetion_data(data):
    x=[]
    y=[]
    for i in range(60, len(data)):
      x.append(data[i-60:i,0])  #1,2  
      y.append(data[i,0])
      # if(i==63):
      #   print(x)
      #   print(y)
      #   print()

    #convert the dataset into numpy array
    x, y= np.array(x), np.array(y)
    
    #reshape the data into three dimentional
    x= np.reshape(x, (x.shape[0], x.shape[1], 1))
    
    return x,y

  x_train,y_train= preparetion_data(train_data)
  x_test,y_test= preparetion_data(testing_data)

  import tensorflow
  model= tensorflow.keras.models.Sequential()

  model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1],1)))
  #model.add(Dropout(.2))

  model.add(LSTM(50, return_sequences=False))

  model.add(Dense(25))
  #model.add(Dense(25,activation='relu'))

  model.add(Dense(1))

  opt= tf.keras.optimizers.Adam(lr=1e-3, decay= 1e-5)

  model.compile(loss="mean_squared_error",optimizer=opt)

  x=model.fit(x_train,y_train, batch_size=1, epochs=1)

  from keras.models import load_model

  model.save('model_pickle.h5')  # creates a HDF5 file 'my_model.h5'
  del model  # deletes the existing model

  # returns a compiled model
  #identical to the previous one
  model = tf.keras.models.load_model(f'{STOCK_NAME}')

  prediction= model.predict(x_test)
  prediction = scaler.inverse_transform(prediction)
  rmse=np.sqrt(np.mean(((prediction- y1_test)**2)))
  rmse


  train=data[:train_len]
  valid=data[train_len:]
  valid['Prediction']= prediction
  plt.figure(figsize=(16,8))
  plt.title("Model")
  plt.xlabel("Date",fontsize=18)
  plt.ylabel("Close Price in Rs.",fontsize=18)
  plt.plot(train['Close'])
  plt.plot(valid[['Close','Prediction']])
  plt.legend(['Train','Val','Prediction',],loc='upper right')
  plt.show()  



#********************************************************************************************
welcome=Label(win, text="welcome to the Stock market predictor", font="Homoarakhn 20 bold")
welcome.grid(row=0,column=10)

Stocks_label=Label(win,text="Stock", font="Homoarakhn 15 bold")
Stocks_label.grid(row=1,column=10)

stocks= StringVar()
stockentry= Entry(win,  textvariable=stocks)
stockentry.grid(row=1,column=11)
 
submit_button= Button(win, text="submit", command=submit)
submit_button.grid(row=2,column=10)

win.mainloop()