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

data=pd.read_csv('C:/Users/BDC/Desktop/PHM08.csv',index_col=None,encoding='big5')



class raw_data_transform:
    
    def __init__(self,data,nb_engines):
          self.data=data
          self.engine=nb_engines
    ###seperate engines   
    def yield_engines(self):
        for i in range(1,self.engine):
            print('現在是第',i,'個engine\n')
            engines.append(data[data['id']==i])
    ###calcuate RUL        
    def auto_RUL(self):
        for j in range(len(engines)):
            engines[j]=engines[j].assign(AUTO_RUL=max(engines[j]['cycle'])-engines[j]['cycle'])
    ###transorm data to 3-dimenstion       
    def twod_to_3d(self):
        for jj in range(0,len(engines)):
            print('-----第',jj+1,'個engine執行中-----\n')
            for ii in range(0,engines[jj].shape[0]):
                    data2d=[]
                    if(np.array(engines[jj]['cycle'])[ii]-100<0):
                        zero_paddings=abs(np.array(engines[jj]['cycle'])[ii]-100)
                        real_data=engines[jj].iloc[0:np.array(engines[jj]['cycle'])[ii],2:26].values
                        zero_data=np.zeros((zero_paddings,24),dtype=float)    
                        data_2d=np.concatenate((zero_data,real_data),axis=0)
                        data_3d.append(data_2d)
                    if(np.array(engines[jj]['cycle'])[ii]-100>=0):
                        start=(np.array(engines[jj]['cycle'])[ii]-100)
                        data_2d=engines[jj].iloc[start:(np.array(engines[jj]['cycle'])[ii]),2:26].values
                        data_3d.append(data_2d)

###define engines as global list
engines=[]
data_3d=[]
data_2d=[]
cla=raw_data_transform(data,219)
cla.yield_engines()
cla.twod_to_3d()
