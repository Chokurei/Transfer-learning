# -*- coding: utf-8 -*-
'''Transfer learning toy example:
1- Train a simple convnet on the MNIST dataset the first 5 digits [0..4].
2- Freeze convolutional layers and fine-tune dense layers
   for the classification of digits [5..9].
Run on GPU: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python mnist_transfer_cnn.py
Get to 99.8% test accuracy after 5 epochs
for the first five digits classifier
and 99.2% for the last five digits after transfer + fine-tuning.
'''

from __future__ import print_function
import numpy as np
import datetime
import os

np.random.seed(1337)  # for reproducibility
from sklearn.metrics import confusion_matrix
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.models import model_from_json

script_name = os.path.basename(__file__)
result_file = os.path.join('..','result')
model_file = os.path.join('..','model')
data_file = os.path.join('..','data')
now_global = datetime.datetime.now()
file_time_global=str(now_global.strftime("%Y-%m-%d-%H-%M"))
#now_global=datetime.datetime.now()
#file_time_global=str(now_global.strftime("%Y-%m-%d-%H-%M"))

now = datetime.datetime.now

batch_size = 128
nb_classes = 5
nb_epoch = 10
# new data set ratio
ratio = 10
# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = 2
# convolution kernel size
kernel_size = 3


if K.image_dim_ordering() == 'th':
    input_shape = (1, img_rows, img_cols)
else:
    input_shape = (img_rows, img_cols, 1)

def feature_deform(feature):
    Feature = feature.reshape(((feature.shape[0],) + input_shape))
    Feature = Feature.astype('float32')
    Feature /= 255
    return Feature

def label_deform(label):
    Y_train = np_utils.to_categorical(label, nb_classes)
    return Y_train        

def train_model(model, train, test, nb_classes):
    X_train = train[0].reshape((train[0].shape[0],) + input_shape)
    X_test = test[0].reshape((test[0].shape[0],) + input_shape)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(train[1], nb_classes)
    Y_test = np_utils.to_categorical(test[1], nb_classes)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    t = now()
    model.fit(X_train, Y_train,
              batch_size=batch_size, nb_epoch=nb_epoch,
              verbose=1,
              validation_data=(X_test, Y_test))
    print('Training time: %s' % (now() - t))
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    return model,(now() - t),score[0],score[1]

def save_model(model,mode):
    json_string=model.to_json()
    if not os.path.isdir(model_file):
        os.mkdir(model_file)
    else:
        if mode == 'ori':
            modle_path=os.path.join(model_file,'ori_'+'architecture'+'_'+file_time_global+'.json')
            open(modle_path,'w').write(json_string)
            model.save_weights(os.path.join(model_file,'ori_'+'model_weights'+'_'+ file_time_global+'.h5'),overwrite= True )
        else:
            modle_path=os.path.join(model_file,'tran_'+'architecture'+'_'+file_time_global+'.json')
            open(modle_path,'w').write(json_string)
            model.save_weights(os.path.join(model_file,'tran_'+'model_weights'+'_'+ file_time_global+'.h5'),overwrite= True )

    
def read_model(mode):
    if mode == 'ori':
        model=model_from_json(open(os.path.join(model_file,'ori_'+'architecture'+'_'+ file_time_global+'.json')).read())
        model.load_weights(os.path.join(model_file,'ori_'+'model_weights'+'_'+file_time_global+'.h5'))
    else:
        model=model_from_json(open(os.path.join(model_file,'tran_'+'architecture'+'_'+ file_time_global+'.json')).read())
        model.load_weights(os.path.join(model_file,'tran_'+'model_weights'+'_'+file_time_global+'.h5'))        
    return model

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# create two datasets one with digits below 5 and one with 5 and above
X_train_lt5 = X_train[y_train < 5]
y_train_lt5 = y_train[y_train < 5]
X_test_lt5 = X_test[y_test < 5]
y_test_lt5 = y_test[y_test < 5]

X_train_gte5 = X_train[y_train >= 5]
y_train_gte5 = y_train[y_train >= 5] - 5  # make classes start at 0 for

train_len = len(X_train_gte5)

X_train_gte5_chosen = X_train_gte5[np.arange(1,train_len,ratio)]
y_train_gte5_chosen = y_train_gte5[np.arange(1,train_len,ratio)] 

X_test_gte5 = X_test[y_test >= 5]         # np_utils.to_categorical
y_test_gte5 = y_test[y_test >= 5] - 5

test_len = len(X_test_gte5)

X_test_gte5_chosen = X_test_gte5[np.arange(1,test_len,ratio)]
y_test_gte5_chosen = y_test_gte5[np.arange(1,test_len,ratio)] 

# define two groups of layers: feature (convolutions) and classification (dense)
feature_layers = [
    Convolution2D(nb_filters, kernel_size, kernel_size,
                  border_mode='valid',
                  input_shape=input_shape),
    Activation('relu'),
    Convolution2D(nb_filters, kernel_size, kernel_size),
    Activation('relu'),
    MaxPooling2D(pool_size=(pool_size, pool_size)),
    Dropout(0.25),
    Flatten(),
]
classification_layers = [
    Dense(128),
    Activation('relu'),
    Dropout(0.5),
    Dense(nb_classes),
    Activation('softmax')
]

# create complete model
model = Sequential(feature_layers + classification_layers)
#############################training#######################################
# train model for 5-digit classification [0..4]
ori_model,ori_time,ori_sco,ori_acc=train_model(model,
            (X_train_lt5, y_train_lt5),
            (X_test_lt5, y_test_lt5), nb_classes)

save_model(ori_model,'ori')

# freeze feature layers and rebuild model
for l in feature_layers:
    l.trainable = False

# transfer: train dense layers for new classification task [5..9]
tran_model,tran_time,tran_sco,tran_acc=train_model(model,
            (X_train_gte5_chosen, y_train_gte5_chosen),
(X_test_gte5_chosen, y_test_gte5_chosen), nb_classes)

save_model(tran_model,'tran')
##############################testing#######################################
############################data preparation#######################
#lt5
X_lt5 = np.concatenate((feature_deform(X_train_lt5),feature_deform(X_test_lt5)),axis=0)
y_lt5 = np.concatenate((y_train_lt5,y_test_lt5),axis=0)
#gte5
X_gte5 = np.concatenate((feature_deform(X_train_gte5),feature_deform(X_test_gte5)),axis=0)
y_gte5 = np.concatenate((y_train_gte5,y_test_gte5),axis=0)
#all
X_all = np.concatenate((X_lt5,X_gte5))
y_all =np.concatenate((y_lt5,y_gte5))
#########################test 0::4############################
#original model
ori_model = read_model('ori')
pre_lt5_ori = ori_model.predict(X_lt5, batch_size=128, verbose=1)
pre_lt5_ori = np.argmax(pre_lt5_ori,axis=1)
ori_con_lt5 = confusion_matrix(y_lt5,pre_lt5_ori)
print('\n'+'original model test lt5:')
print('Confusion matrix:')
print(ori_con_lt5)
print('Accuracy:')
print(ori_con_lt5.trace()/np.sum(ori_con_lt5[::]))
#transfered model
tran_model = read_model('tran')
pre_lt5_tran = tran_model.predict(X_lt5, batch_size=128, verbose=1)
pre_lt5_tran = np.argmax(pre_lt5_tran,axis=1)
tran_con_lt5 = confusion_matrix(y_lt5,pre_lt5_tran)
print('\n'+'transfer model test lt5:')
print('Confusion matrix:')
print(tran_con_lt5)
print('Accuracy:')
print(tran_con_lt5.trace()/np.sum(tran_con_lt5[::]))
######################test 5::9##############################
#original model
ori_model = read_model('ori')
pre_gte5_ori = ori_model.predict(X_gte5, batch_size=128, verbose=1)
pre_gte5_ori = np.argmax(pre_gte5_ori,axis=1)
ori_con_gte5 = confusion_matrix(y_gte5,pre_gte5_ori)
print('\n'+'original model test gte5:')
print('Confusion matrix:')
print(ori_con_gte5)
print('Accuracy:')
print(ori_con_gte5.trace()/np.sum(ori_con_gte5[::]))
#transfered model
tran_model = read_model('tran')
pre_gte5_tran = tran_model.predict(X_gte5, batch_size=128, verbose=1)
pre_gte5_tran = np.argmax(pre_gte5_tran,axis=1)
tran_con_gte5 = confusion_matrix(y_gte5,pre_gte5_tran)
print('\n'+'transfer model test gte5:')
print('Confusion matrix:')
print(tran_con_gte5)
print('Accuracy:')
print(tran_con_gte5.trace()/np.sum(tran_con_gte5[::]))
########################test all: 0::9########################
#original model
ori_model = read_model('ori')
pre_all_ori = ori_model.predict(X_all, batch_size=128, verbose=1)
pre_all_ori = np.argmax(pre_all_ori,axis=1)
ori_con_all = confusion_matrix(y_all,pre_all_ori)
print('\n'+'original model test all:')
print('Confusion matrix:')
print(ori_con_all)
print('Accuracy:')
print(ori_con_all.trace()/np.sum(ori_con_all[::]))
#transfer model
tran_model = read_model('tran')
pre_all_tran = tran_model.predict(X_all, batch_size=128, verbose=1)
pre_all_tran = np.argmax(pre_all_tran,axis=1)
tran_con_all = confusion_matrix(y_all,pre_all_tran)
print('\n'+'transfer model test all:')
print('Confusion matrix:')
print(tran_con_all)
print('Accuracy:')
print(tran_con_all.trace()/np.sum(tran_con_all[::]))

def log_in():
    #######################################log#####################################
    log_file=open(os.path.join(result_file,'my_log.txt'),'a')
    log_file.write('########################Time:'+file_time_global+'########################'+'\n')
    log_file.write('#                    File:'+script_name+'\n')
    log_file.write('######################Transfer data ratio: 1/'+str(ratio)+'########################'+'\n')
    log_file.write('Transfer learning trail, data set: MNIST digit identification'+'\n')
    log_file.write('Sample size:'+''+str(img_rows)+' x '+str(img_cols)+'\n')
    log_file.write('+++++++++++++++++training original and transfered model++++++++++++++++'+'\n')
    log_file.write('Origial model:'+'\n')
    log_file.write('Data set: digit 0 ~ 4'+'\n')
    log_file.write('Number of trianing samples:'+str(len(X_train_lt5))+'\n')
    log_file.write('          testing samples:'+str(len(X_test_lt5))+'\n')
    log_file.write('Batch_size:'+str(batch_size)+'     '+'Iteration:'+str(nb_epoch)+'\n')
    log_file.write('Training time:'+str(ori_time)+'    Every_iter:'+str(ori_time/nb_epoch)+'\n')
    log_file.write('Validation accuracy:'+str(ori_acc)+'     Score:'+str(ori_sco)+'\n')
    log_file.write('\n')
    log_file.write('transfered model:'+'\n')
    log_file.write('Data set: digit 5 ~ 9'+'\n')
    log_file.write('Data ratio: 1/'+str(ratio)+'\n')
    log_file.write('Number of trianing samples:'+str(len(X_train_gte5_chosen))+'\n')
    log_file.write('          testing samples:'+str(len(X_test_gte5_chosen))+'\n')
    log_file.write('Batch_size:'+str(batch_size)+'     '+'Iteration:'+str(nb_epoch)+'\n')
    log_file.write('Training time:'+str(tran_time)+'    Every_iter:'+str(tran_time/nb_epoch)+'\n')
    log_file.write('Validation accuracy:'+str(tran_acc)+'     Score:'+str(tran_sco)+'\n')
    log_file.write('+++++++++++++++++testing original and transfered model++++++++++++++++'+'\n')
    log_file.write('-----------------------group 1: testing 0::4-----------------'+'\n')
    log_file.write('Testing data amount: '+str(len(X_train_lt5))+'\n')
    log_file.write('Original model result:'+'\n')
    log_file.write('Confusion matrix:'+'\n')
    log_file.write(str(ori_con_lt5)+'\n')
    log_file.write('Accuracy:'+'\n')
    log_file.write(str(ori_con_lt5.trace()/np.sum(ori_con_lt5[::]))+'\n')
    log_file.write('transfered model result:'+'\n')
    log_file.write('Confusion matrix:'+'\n')
    log_file.write(str(tran_con_lt5)+'\n')
    log_file.write('Accuracy:'+'\n')
    log_file.write(str(tran_con_lt5.trace()/np.sum(tran_con_lt5[::]))+'\n')
    log_file.write('-----------------------group 2: testing 5::9-----------------'+'\n')
    log_file.write('Testing data amount: '+str(len(X_train_gte5))+'\n')
    log_file.write('Original model result:'+'\n')
    log_file.write('Confusion matrix:'+'\n')
    log_file.write(str(ori_con_gte5)+'\n')
    log_file.write('Accuracy:'+'\n')
    log_file.write(str(ori_con_gte5.trace()/np.sum(ori_con_gte5[::]))+'\n')
    log_file.write('transfered model result:'+'\n')
    log_file.write('Confusion matrix:'+'\n')
    log_file.write(str(tran_con_gte5)+'\n')
    log_file.write('Accuracy:'+'\n')
    log_file.write(str(tran_con_gte5.trace()/np.sum(tran_con_gte5[::]))+'\n')
    log_file.write('-----------------------group 3: testing 0::9-----------------'+'\n')
    log_file.write('Testing data amount: '+str(len(X_all))+'\n')
    log_file.write('Original model result:'+'\n')
    log_file.write('Confusion matrix:'+'\n')
    log_file.write(str(ori_con_all)+'\n')
    log_file.write('Accuracy:'+'\n')
    log_file.write(str(ori_con_all.trace()/np.sum(ori_con_all[::]))+'\n')
    log_file.write('transfered model result:'+'\n')
    log_file.write('Confusion matrix:'+'\n')
    log_file.write(str(tran_con_all)+'\n')
    log_file.write('Accuracy:'+'\n')
    log_file.write(str(tran_con_all.trace()/np.sum(tran_con_all[::]))+'\n')
    log_file.write('\n')
    log_file.close()

log_in()
