# k-fold activation
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import StratifiedKFold
import numpy
import matplotlib.pyplot as plt
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
X=np.array(end)[:,2:-1]
Y=np.array(end)[:,-1]
X=X.reshape((X.shape[0],1,X.shape[1]))
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
kfold_loss=[]
for i in (activation):
    print('activation =',i)
    loss=[]
    for train, test in kfold.split(X, Y):
        model = Sequential()
        model.add(LSTM(50, input_shape=(1,24),return_sequences=False))
        model.add(Activation(i))
    #model.add(Dropout(0.5))
    #model.add(Activation('relu'))
    #model.add(Dense(100))
    #model.add(Activation('relu'))
    #model.add(Dense(100))
        model.add(Dense(1))
        ad=optimizers.Adam(lr=0.001)
        model.compile(loss='mean_squared_error', optimizer=ad, metrics=['mse'])
        history=model.fit(X[train], Y[train], epochs=100,batch_size=128, verbose=0,validation_data=(X[test], Y[test]))
        loss.append(history.history['loss'])
    kfold_loss.append(numpy.mean(loss, axis=0))



#k-fold optimizers
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import StratifiedKFold
import numpy
import matplotlib.pyplot as plt
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
X=np.array(end)[:,2:-1]
Y=np.array(end)[:,-1]
X=X.reshape((X.shape[0],1,X.shape[1]))
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
kfold_loss=[]
for i in (optimizer ):
    print('optimizer  =',i)
    loss=[]
    for train, test in kfold.split(X, Y):
        model = Sequential()
        model.add(LSTM(50, input_shape=(1,24),return_sequences=False))
        model.add(Activation('relu'))
    #model.add(Dropout(0.5))
    #model.add(Activation('relu'))
    #model.add(Dense(100))
    #model.add(Activation('relu'))
    #model.add(Dense(100))
        model.add(Dense(1))
        #optimizers.(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='mean_squared_error', optimizer=i, metrics=['mse'])
        history=model.fit(X[train], Y[train], epochs=100,batch_size=128, verbose=0,validation_data=(X[test], Y[test]))
        loss.append(history.history['loss'])
    kfold_loss.append(numpy.mean(loss, axis=0))
    
    
    
    
#k-fold hiddenlayers 
from keras.callbacks import EarlyStopping
import time
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import StratifiedKFold
import numpy
import matplotlib.pyplot as plt
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
X=np.array(end)[:,2:-1]
Y=np.array(end)[:,-1]
X=X.reshape((X.shape[0],1,X.shape[1]))
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
kfold_loss=[]
kfold_time=[]
kfold_epoch1=[]
batch=[32,64,128]
lstm1=[32,64,32,32,96,96]
lstm2=[32,64,64,64,96,96]
nn1=[8,8,8,16,8,16]
nn2=[8,8,8,16,8,16]

for qq in batch:
    print('batch_size:',qq)
    for i,j,k,l in zip(lstm1,lstm2,nn1,nn2):
        print(i,j,k,l) 
        rmse=[]
        epoch1=[]
        time1=[]
        for train, test in kfold.split(X, Y):
            model = Sequential()
            model.add(LSTM(i, input_shape=(1,24),return_sequences=True))
            model.add(Activation('relu'))
            model.add(LSTM(j,return_sequences=False))
            model.add(Activation('relu'))
            model.add(Dense(k))
            model.add(Activation('relu'))
            model.add(Dense(l))
            model.add(Activation('relu'))
            model.add(Dense(1))
        #optimizers.(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
            model.compile(loss='mean_squared_error', optimizer='nadam', metrics=['mse'])
            earlystop = [EarlyStopping(patience=10,monitor='val_loss')]

            start = time.time()
        
            history=model.fit(X[train], Y[train], epochs=1000,callbacks=earlystop,batch_size=qq,verbose=0,validation_data=(X[test], Y[test]))
            end = time.time()
            time1.append(end-start)
            rmse.append(sqrt(history.history['val_loss'][max(history.epoch)-10]))
            epoch1.append(max(history.epoch)-10)
        kfold_loss.append(numpy.mean(rmse, axis=0))
        kfold_time.append(numpy.mean(time1, axis=0))
        kfold_epoch1.append(numpy.mean(epoch1, axis=0))
