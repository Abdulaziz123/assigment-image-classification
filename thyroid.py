import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
import tensorflow as tf
import json
plt.style.use("ggplot")
from itertools import chain
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import classification_report
from keras import regularizers
from keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras.layers import Dropout, Input
from tensorflow.python.keras.layers import MaxPooling1D
from keras.utils import np_utils
from tensorflow.keras.metrics import SpecificityAtSensitivity, SensitivityAtSpecificity, BinaryAccuracy, Accuracy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam as Adam
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

import tensorflow.keras.backend as K

#data preprocessing
df_dados=pd.read_csv('df_clipped.csv')
malig=np.ones(len(df_dados),dtype=int)
malig[np.where(df_dados['tirads'].values=='2')]=0
malig[np.where(df_dados['tirads'].values=='3')]=0
df_dados['malig']=malig

df_dados.head()
gray_three=[]
for i in range (452):
        image = cv2.imread('clipped/'+str(i)+'.png')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_three1 = (cv2.merge([gray,gray,gray]))
        gray_three.append(gray_three1)

np.shape(gray_three)

nimg=len(gray_three)
nx=224 #np.shape(gray_three)[1] #dimension x
ny=224 #np.shape(gray_three)[2] #dimension y

X=np.zeros(((nimg,nx,ny,3)))
for id_ in range (len(gray_three)):
    x_img = gray_three[id_]
    x_img = resize(x_img, (nx, ny,3), mode = 'constant', preserve_range = True)    
    X[id_,:,:]=x_img/255

Y=df_dados['malig'].values

#split data for test and train, 20% of data for test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
test_size = (X_test.shape[0]/X_train.shape[0])

weights=class_weight.compute_class_weight('balanced', np.unique(Y_train), Y_train)

#differen models that I used: InceptionV3; ResNet50V2
base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(nx,ny,3))
base_model.trainable = False

#load json type data
conf = json.loads(base_model.to_json())

for l in conf['config']['layers']:
    if l['class_name'] == 'BatchNormalization':
        l['config']['momentum'] = 0.90
        l['config']['trainable']= True

m = base_model.from_config(conf['config'])
for l in base_model.layers:
    m.get_layer(l.name).set_weights(l.get_weights())

base_model = m

#network structure
model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())#pooling
model.add(BatchNormalization())#batch normalization
model.add(Dropout(0.4)) #dropout
model.add(Dense(100, activation='relu')) #regularization 
model.add(Dropout(0.4))#dropout
model.add(Dense(1, activation='sigmoid'))#activation function

#Compile model
epochs = 50
adam = Adam(lr=0.0001)#learning_rate = 0.0001
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['acc'])
tf.random.set_seed(2)

#Evaluation
history=model.fit(X_train, Y_train, epochs=epochs, validation_data=(X_test, Y_test), batch_size=32)

_,train_acc = model.evaluate(X_train, Y_train)
_,test_acc = model.evaluate(X_test, Y_test)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc )) 

training_accuracy = history.history['acc']
test_accuracy = history.history['val_acc']

# Create count of the number of epochs
epoch_count = range(1, len(training_accuracy) + 1)
# predict probabilities for test set
yhat_probs = model.predict(X_test)
# predict crisp classes for test set
yhat_classes = model.predict_classes(X_test)
# accuracy: (tp + tn) / (p + n)
accuracy = accuracy_score(Y_test, yhat_classes)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = precision_score(Y_test, yhat_classes)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = recall_score(Y_test, yhat_classes)
print('Recall: %f' % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(Y_test, yhat_classes)
print('F1 score: %f' % f1)

# confusion matrix
matrix = confusion_matrix(Y_test, yhat_classes)
# Normalise
cmn = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
# Transform to df for easier plotting
cm_df = pd.DataFrame(cmn,
                     index = ['Benigns','Maligns'], 
                     columns = ['Benigns','Maligns'])

plt.figure(figsize=(5.5,4))
sns.heatmap(cm_df, annot=True)
plt.title('MobileNetV2\nAccuracy:{0:.3f}'.format(accuracy_score(Y_test, yhat_classes)))
plt.ylabel('True label')
plt.xlabel('Predicted label')

# Visualize accuracy history
plt.figure(figsize=(5.5,4))
plt.title('Accuracy')
plt.plot(history.history['acc'], label='train')
plt.plot(history.history['val_acc'], label='test')
plt.legend(['Train Accuracy', 'Test Accuracy'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy Score')
plt.show();

