import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import regex as re
from keras.models import Sequential
from keras.layers import *
from tensorflow.keras.optimizers import Adam 
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
from collections import Counter


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

base_dir = '../../data/'

users = ["user7", "user9", "user12", "user15", "user16", "user20","user21","user23","user29","user35"]


# Set the random seeds
my_seed = 123456
random.seed(my_seed)
np.random.seed(my_seed)
tf.random.set_seed(my_seed)

    
def create_gru_model(shape):
    model = tf.keras.Sequential([ tf.keras.layers.Input(shape=shape) ])
    recurrent_block1 = tf.keras.layers.GRU(200, activation="tanh", return_sequences=True)
    recurrent_block1 = tf.keras.layers.Bidirectional(recurrent_block1)
    
    recurrent_block2 = tf.keras.layers.GRU(200, activation="tanh")
    recurrent_block2 = tf.keras.layers.Bidirectional(recurrent_block2)

    model.add(recurrent_block1)
    model.add(recurrent_block2)
    
    model.add(tf.keras.layers.Dropout(0.20))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    return model

def eval_binary_classifier(y_pred):
    prob_labels = np.array(y_pred).ravel()
    pred_labels = prob_labels > 0.5
    pred_labels = pred_labels.astype(int)
    return pred_labels

# find the user's of the corresponding session that are not illegal
def findSessionLabelIsIllegal(df, session):
    try:
        return (df.loc[df['filename'] == session].is_illegal.values[0])
    except :
        return None



# Define the labels
labels = pd.read_csv('../../public_labels.csv')

def cleanSession(session,user):
    data = pd.read_table( base_dir + str(user)+ "/" + session, sep=',')
    # Drop the first timestamp
    data.drop('record timestamp', axis=1, inplace=True)
    pd.set_option('display.float_format', lambda x: '%.5f' % x)
    data.round(5)
    data.drop("button", axis=1, inplace=True)
    data.drop("state", axis=1, inplace=True)
    return data 

def normalisedOverTime(data):
    # reset the index from 0 and remove the timesteps 
    data_init = data
    data.drop("index", axis=1, inplace=True)
    # Apply the difference numpy function to obtain a dataframe with 3 rows, dt, dx, dy
    # Delta x being the difference between x_{i} and x_{i-1}
    data = data.diff(axis = 0, periods = 1)
    # This function alters the first row with NaN values, since no diff. can be computed
    # for the 1st element in the df.
    # This row will be removed to replace it by the duplicate of the second element. 
    data.drop(0,axis=0,inplace=True)
    data.loc[-1] = data.loc[1]
    data.index = data.index + 1  # shifting index
    data.sort_index(inplace=True)
    data["dx"] = data["x"] / data["client timestamp"]
    data["dy"] = data["y"] / data["client timestamp"]
    data.drop(["x","y", "client timestamp"],axis=1,inplace=True)
    return data 


def getlegalsessionsfromotheruser(user,left):
    global users
    chosenUser = user
    is_legal = 0
    sessions_to_return = []
    X_dataset = []
    while is_legal < left:
        while chosenUser == user:
            chosenUser = random.choice(users)
        sessions = os.listdir(base_dir + str(chosenUser))
        for session in sessions: 
            data = cleanSession(session,chosenUser)
            # get the legal session from the randomly chosen user
            if findSessionLabelIsIllegal(labels, session) == 0:
                i = 0
                if data[i:i+300].shape[0] < 300 :
                    break  
                if is_legal >= left :
                    break
                else :
                    is_legal += 1
                    data_tmp  = data[i:i+300].reset_index()
                    # reset the index from 0 and remove the timesteps
                    data_tmp = normalisedOverTime(data_tmp)
                    sessions_to_return.append(data_tmp)
                    X_dataset.append(np.array(data_tmp))
                    i += 300
    
    return X_dataset

def rawCoordinates(data):
    data.drop("index", axis=1, inplace=True)
    return data

def createDataset(user):
    X_dataset = []
    Y_dataset = []
    is_legal = 1
    is_illegal = 1
    pd.set_option('display.float_format', lambda x: '%.5f' % x)
    sessions = os.listdir(base_dir+str(user))
    for session in sessions:
        # only take into account legal sessions_by_users
        if findSessionLabelIsIllegal(labels, session) == 1:
            data = cleanSession(session,user)
            i = 0
            while True :
                if is_illegal > 10000 :
                    break
                if data[i:i+300].shape[0] < 300 :
                    break  
                else :
                    is_illegal += 1
                    data_tmp  = data[i:i+300].reset_index()
                    # reset the index from 0 and remove the timesteps
                    data_tmp = normalisedOverTime(data_tmp)
                    X_dataset.append(np.array(data_tmp))
                    Y_dataset.append(1)
                    i += 300
        else :
            previous_data = pd.DataFrame()
            while True:
                i = 0
                data = cleanSession(session,user)
                if data[i:i+300].shape[0] < 300 :
                    break
                else :
                    is_legal += 1
                    data_tmp  = data[i:i+300].reset_index()
                    # reset the index from 0 and remove the timesteps
                    data_tmp = normalisedOverTime(data_tmp)
                    i += 300
                    X_dataset.append(np.array(data_tmp))
                    Y_dataset.append(0)
                    if previous_data.equals(data_tmp):
                        break
                    else :
                        previous_data = data_tmp


    print(Counter(Y_dataset))
    numberoflegalinput = Counter(Y_dataset)[0]
    print(is_illegal)
    randomnegativesamples = getlegalsessionsfromotheruser(user, numberoflegalinput-is_illegal)
    for sess in randomnegativesamples:
        X_dataset.append(sess)
        Y_dataset.append(1)
    print(Counter(Y_dataset))
    return X_dataset, Y_dataset


for user in users:
    # 5-fold validation to record the correct metrics
    for fold in range(5):
        NAME = "{}-fold-{}".format(user,fold+1)
        X_dataset, Y_dataset = createDataset(user)
        # Divice into training and testing set
        X_train, X_test, Y_train, Y_test = train_test_split(X_dataset, Y_dataset, test_size=0.1, random_state=42, stratify=Y_dataset)
        Y_train = tf.reshape(Y_train, [len(Y_train), 1])
        Y_test = tf.reshape(Y_test, [len(Y_test), 1])
        # X_train consists of a list of lists with 100 elements, serving as input to the 100 input neuronsbs    
        X_test = np.array(X_test, dtype=int)
        X_train = np.array(X_train, dtype=int)
        X_test = np.array(X_test, dtype=int)
        model = None
        model = create_gru_model(X_train[0].shape)
        model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0001), metrics=['accuracy'])
        tf.keras.utils.plot_model(
        model,
        to_file='final-model.png',
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
        rankdir='TB',
        expand_nested=True,
        dpi=96,
        layer_range=None,
        show_layer_activations=True
        )   
        model.summary()
        model.fit(X_train, Y_train, epochs = 400, batch_size =150, verbose = 1, shuffle = True, validation_split=0.1, callbacks = [tf.keras.callbacks.EarlyStopping(patience=40, verbose=1,restore_best_weights=True, monitor='val_loss', mode='auto'),tf.keras.callbacks.TensorBoard("models-Tensorboard/{}".format(NAME), profile_batch=0),])
        # Save model once training is done.
        filename = str(user) + '-fold' + str(fold+1)
        model_file = os.path.join("balanced-models", '{}.h5'.format(filename))
        model.save(model_file)
        # Save the testing dataset into a txt file to be used for correct validation 
        np.save(os.path.join("models-allusers-testingsets", filename + "-test-X"), X_test)
        np.save(os.path.join("models-allusers-testingsets", filename + "-test-Y"), Y_test)
        print('Model saved as {}'.format(model_file))
        print('Testing Set saved in a numpy file.')

