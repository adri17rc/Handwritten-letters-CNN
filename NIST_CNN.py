#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 09:54:51 2022

@author: adrianromero
"""

#NIST DATASET. Handwritten letters recognition 

import numpy as np
import pandas as pd
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, add, Dense, MaxPooling2D, BatchNormalization, Flatten
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

#Import NIST dataset. It is ordered, therefore it has to be shuffled before using it.

data_path='/Volumes/Toshiba-ext/archive/A_Z Handwritten Data.csv'

data=pd.read_csv(data_path)

print(data.shape)
#Size: 372450, 785. 785=784(28x28)+1. Last column, y values?
data=shuffle(data)

print(data.head())

data.rename(columns={'0':'label'}, inplace=True)

# Split data the X - Our data , and y - the prdict label
X = data.drop('label',axis = 1)
y = data['label']


#First 9 images 
rows=3
columns=3
plt.figure(figsize = (12,10))
for i in range(9):
    
    plt.subplot(columns,rows,i+1) 
    plt.imshow(X.iloc[i].values.reshape(28,28),cmap='Greys')
    
plt.show()

#Normalize x data (pictures)

standard_scaler = MinMaxScaler()
standard_scaler.fit(X) #Defines max and min to scale. Chose row 1

X = standard_scaler.transform(X)


#Reshape x values in arrays of 28x28 (MNIST size)
X=X.reshape((X.shape[0], 28, 28, 1))

#Plot the number of letters of each type. Originally defined by their position in the alphabet

alphabet={0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X',24:'Y',25:'Z'}

#Redefine y 
data['label']=data['label'].map(alphabet)


label_size = data.groupby('label').size()
label_size.plot.barh(figsize=(10,10),cmap='hsv')

plt.show()

#Y to categorical 
y=keras.utils.to_categorical(y)


#Split data in train and test set 

trainy=y[:300000]
testy=y[300000:]
trainx=X[:300000,:,:,:]
testx=X[300000:,:,:,:]



#Setup the CNN

model=Sequential()
 
model.add(Conv2D(32,(3,3),activation='relu',kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(90,activation='relu'))
model.add(Dense(26,activation='softmax'))

model.summary()

model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])

results=model.fit(trainx,trainy,epochs=8,batch_size=200,validation_data=(testx,testy))

#Plots for loss and acuracy 

#Cross entropy loss 
plt.plot(results.history['loss'], label='train')
plt.plot(results.history['val_loss'], label='test')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

#Accuracy
plt.plot(results.history['accuracy'], label='train')
plt.plot(results.history['val_accuracy'], label='test')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['Train', 'Test'], loc='upper left')

plt.show()

#Predict 100 letters (first 100 in train_x)

testx_2=trainx[:100]
testy_2=trainy[:100]

prediction=model.predict(testx_2)

num_pos=np.zeros(len(testx_2),dtype=int)
letters='' 

for i in range(len(testx_2)):
    num_pos[i]=np.argmax(prediction[i])
    
    if num_pos[i]==0:
        letters=letters+'\n'+'A'
    elif num_pos[i]==1:
        letters=letters+'\n'+'B'
    elif num_pos[i]==2:
        letters=letters+'\n'+'C'
    elif num_pos[i]==3:
        letters=letters+'\n'+'D'
    elif num_pos[i]==4:
        letters=letters+'\n'+'E'
    elif num_pos[i]==5:
        letters=letters+'\n'+'F'
    elif num_pos[i]==6:
        letters=letters+'\n'+'G'
    elif num_pos[i]==7:
        letters=letters+'\n'+'H'
    elif num_pos[i]==8:
        letters=letters+'\n'+'I'
    elif num_pos[i]==9:
        letters=letters+'\n'+'J'
    elif num_pos[i]==10:
        letters=letters+'\n'+'K'
    elif num_pos[i]==11:
        letters=letters+'\n'+'L'
    elif num_pos[i]==12:
        letters=letters+'\n'+'M'
    elif num_pos[i]==13:
        letters=letters+'\n'+'N'
    elif num_pos[i]==14:
        letters=letters+'\n'+'L'
    elif num_pos[i]==15:
        letters=letters+'\n'+'O'
    elif num_pos[i]==16:
        letters=letters+'\n'+'P'
    elif num_pos[i]==17:
        letters=letters+'\n'+'Q'
    elif num_pos[i]==18:
        letters=letters+'\n'+'R'
    elif num_pos[i]==19:
        letters=letters+'\n'+'S'
    elif num_pos[i]==20:
        letters=letters+'\n'+'T'
    elif num_pos[i]==21:
        letters=letters+'\n'+'U'
    elif num_pos[i]==22:
        letters=letters+'\n'+'V'
    elif num_pos[i]==23:
        letters=letters+'\n'+'W'
    elif num_pos[i]==24:
        letters=letters+'\n'+'X'
    elif num_pos[i]==25:
        letters=letters+'\n'+'Y'
    elif num_pos[i]==26:
        letters=letters+'\n'+'Z'
        
print(letters)
  

#Translate predictions to a string of letters 








