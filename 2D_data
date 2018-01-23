import os
import numpy as np
import pandas as pd
import glob
import re
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
from sklearn.preprocessing import normalize
from keras.models import load_model # keras.models.load_model(filepath)
from keras import optimizers
from keras.utils import plot_model
import pydot
from sklearn.preprocessing import scale
from keras.layers import Dropout
###read data
data=pd.read_csv('C:/Users/huang/Desktop/PHM08.csv',index_col=None,encoding='big5')
def normalize_data(data,col): # normalize
    return((data[:,col]-data[:,col].min())/(data[:,col].max()-data[:,col].min()))
def denormalize_data(nor_data,raw_data,col):
    return((nor_data[:]*(raw_data[:,col].max()-raw_data[:,col].min()))+raw_data[:,col].min())

data1=scale(data.iloc[:,2:26])
end=data.copy()
end.iloc[:,2:26]=data1
###create 2d data
np.array(end)[:40000,2:]
train_X, train_y = np.array(end)[:40000,2:], data.values[:40000,-1] #ctun 2016
test_X, test_y = np.array(end)[40000:,2:], data.values[40000:,-1]
###reshape to lstm format
train_X = train_X.reshape((train_X.shape[0],1,train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0],1,test_X.shape[1]))
#train_X = train_X.reshape((train_X.shape[0],train_X.shape[1],1))
#test_X = test_X.reshape((test_X.shape[0],test_X.shape[1],1))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
###lstm
model = Sequential()
model.add(LSTM(48, input_shape=(1,25),return_sequences=True))
model.add(Activation('relu'))
model.add(LSTM(48))
model.add(Activation('relu'))
# model.add(LSTM())
# model.add(Activation('relu'))
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(100))
# model.add(Dropout(0.5))
# model.add(Activation('relu'))
# model.add(Dense(100))
# model.add(Dropout(0.5))
# model.add(Activation('relu'))
model.add(Dense(1))
ad=optimizers.Adam(lr=0.001)
model.compile(loss='mean_squared_error', optimizer=ad, metrics=['mse'])
#model.compile(optimizer = "rmsprop", loss = "root_mean_squared_error")#, metrics =["accuracy"])
history=model.fit(train_X, train_y, epochs=10, batch_size=128,validation_data=(test_X, test_y),verbose=2, shuffle=False)
aa=model.predict(test_X)
###validation
from sklearn.metrics import mean_squared_error
from math import sqrt
bb=np.around(aa)
bb=bb.reshape(5918)
bb[0:100]
### rmse
rms = sqrt(mean_squared_error(test_y,bb))
rms
###plot
pyplot.plot(bb)
pyplot.ylabel("RUL")
pyplot.show()
pyplot.plot(test_y,label="TRUE")
pyplot.ylabel("RUL")
#pyplot.ylim((0,500) ) 
pyplot.show()
###Autoencoder
from keras.layers import Input, Dense
from keras.models import Model
# this is the size of our encoded representations
encoding_dim = 10  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
# this is our input placeholder
input_img = Input(shape=(24,))
# "encoded" is the encoded representation of the input
encoded = Dense(15, activation='relu')(input_img)
encoded=Dense(10, activation='relu')(encoded)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(10, activation='relu')(encoded)
decoded = Dense(15, activation='relu')(decoded)
decoded = Dense(24, activation='tanh')(decoded)
# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)


encoder = Model(input=input_img, output=encoded) #data reduction

autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.fit(data1, data1,
                nb_epoch=50,
                batch_size=128,
                )
###predict               
autodata=encoder.predict(data1)
autodata=scale(autodata)
autodata=pd.DataFrame(autodata)
auto=pd.concat([data,autodata],axis=1)
AUTO=auto.drop(auto.columns[2:27],axis=1)
AUTO
###save & load model
from keras.models import load_model
#model.save('C:/Users/BDC/Desktop/lstm4848.h5')
model = load_model('C:/Users/huang/Desktop/model/lstm4848.h5')
aa=model.predict(test_X)
