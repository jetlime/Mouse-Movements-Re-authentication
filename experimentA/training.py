#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Paul Houssel
# License: MIT License

"""
This python script serves to train the models for every user
with the distribution of the experiment A.
"""

from pandas import set_option,read_csv, read_table, DataFrame
from os import environ, listdir, path, mkdir
from numpy import random, save, array
from random import seed

# Import from the ML framework
from tensorflow import reshape
from tensorflow.random import set_seed
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.layers import GRU, Bidirectional, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard


# Ignore the Tensorflow Informations and Warning
environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Directory of the Dataset
base_dir = '../data/'

# Define the labels
labels = read_csv('../public_labels.csv')

# List of all the users in the dataset
users = ["user7", "user9", "user12", "user15", "user16", "user20","user21","user23","user29","user35"]

# The experimental results of the final report, were
# obtained with the following seed
my_seed = 123456
# Set python random seed
seed(my_seed)
# Set Numpy random seed
random.seed(my_seed)
# Set tensorflow random seed
set_seed(my_seed)

# Implementation of the Classifier Model
def createGruModel(shape):
    model = Sequential([ Input(shape=shape)])
    recurrent_block1 = GRU(200, activation="tanh", return_sequences=True)
    recurrent_block1 = Bidirectional(recurrent_block1)

    recurrent_block2 = GRU(200, activation="tanh")
    recurrent_block2 = Bidirectional(recurrent_block2)

    model.add(recurrent_block1)
    model.add(recurrent_block2)

    model.add(Dropout(0.20))
    model.add(Dense(1, activation='sigmoid'))
    return model

# Fetch the session file, to transform it into a modifible dataframe
def cleanSession(session,user):
    data = read_table(path.join(base_dir, str(user), session), sep=',')
    # Convert and round up the data
    set_option('display.float_format', lambda x: '%.5f' % x)
    data.round(5)
    # Remove the columns not needed by our model
    data.drop("button", axis=1, inplace=True)
    data.drop("state", axis=1, inplace=True)
    data.drop('record timestamp', axis=1, inplace=True)
    return data

# Normalise the mouse coordinates over time
def normalisedOverTime(data):
    # reset the index of the dataframe
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
    # Remove the three columns not needed anymore, the timestamp and the mouse coordinates
    data.drop(["x","y", "client timestamp"],axis=1,inplace=True)
    return data

# find out if the session of the given user is illegal or not
def sessionIsIllegal(df, session):
    try:
        return (df.loc[df['filename'] == session].is_illegal.values[0])
    except :
        return None


# Create the dataset for every user
# Binary Labels
# O: Legal Data ; 1: Illegal Data
def createDataset(user):
    X_dataset = []
    Y_dataset = []
    # Count the number of positive and negative data collected
    is_legal = 0
    is_illegal = 0
    # Round up the float number of the future panda dataframes
    set_option('display.float_format', lambda x: '%.5f' % x)
    # List all the sessions in the dataset for the given user
    sessions = listdir(base_dir+str(user))
    for session in sessions:
        # If the session is Illegal
        if sessionIsIllegal(labels, session) == 1:
            data = cleanSession(session,user)
            i = 0
            while True :
                if is_illegal > 10000 :
                    break
                # If the reminding data has less then 300 timestamps then it is not considered
                if data[i:i+300].shape[0] < 300 :
                    break
                else :
                    # Once the session is considered, 300 timestamps are
                    # stripped apart to create a new entry in our Dataset
                    is_illegal += 1
                    data_tmp  = data[i:i+300].reset_index()
                    # reset the index from 0 and remove the timesteps
                    data_tmp = normalisedOverTime(data_tmp)
                    X_dataset.append(array(data_tmp))
                    # The corresponding label is defined to indicate it is a illegal mouse movement session
                    Y_dataset.append(1)
                    i += 300
        # If the session is legal
        else :
            previous_data = DataFrame()
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
                    X_dataset.append(array(data_tmp))
                    # Definet the legal label
                    Y_dataset.append(0)
                    if previous_data.equals(data_tmp):
                        break
                    else :
                        previous_data = data_tmp
    return X_dataset, Y_dataset



if __name__ == "__main__":
    # Create the needed directories for this experiment
    print("...Creating necessary folders...")
    mkdir("models")
    mkdir("models-testingsets")
    mkdir("models-Tensorboard")
    # For every user, the models will be created and trained with 5 datasets
    # (which each have different training and testing sets)
    # in order to obtain a more widespread view of the model when it comes to 
    # evaluating it
    for user in users:
        # 5-fold validation to record a good overview of the model
        for fold in range(5):
            # Name of the file where the trained model will be stored
            NAME = "{}-fold-{}".format(user,fold+1)
            X_dataset, Y_dataset = createDataset(user)
            # Divice the above created dataset into training and testing set
            X_train, X_test, Y_train, Y_test = train_test_split(X_dataset, Y_dataset, test_size=0.1, random_state=42, stratify=Y_dataset)
            Y_train = reshape(Y_train, [len(Y_train), 1])
            Y_test = reshape(Y_test, [len(Y_test), 1])
            X_test = array(X_test, dtype=int)
            X_train = array(X_train, dtype=int)
            X_test = array(X_test, dtype=int)
            model = None
            model = createGruModel(X_train[0].shape)
            model.compile(loss='binary_crossentropy', optimizer=Adam(0.0001), metrics=['accuracy'])
            # Train the model
            model.fit(X_train, Y_train, epochs = 400, batch_size =150, verbose = 1, shuffle = True, validation_split=0.1, callbacks = [EarlyStopping(patience=40, verbose=1,restore_best_weights=True, monitor='val_loss', mode='auto'),TensorBoard("models-Tensorboard/{}".format(NAME), profile_batch=0),])
            # Save the model once training is done.
            model_file = path.join("models", '{}.h5'.format(NAME))
            model.save(model_file)
            # Save the testing dataset into a txt file to be used for correct validation 
            save(path.join("models-testingsets", NAME + "-test-X"), X_test)
            save(path.join("models-testingsets", NAME + "-test-Y"), Y_test)
            print('Model saved as {}'.format(model_file))
            print('Testing Set saved in a numpy file.')
