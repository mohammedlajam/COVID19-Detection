# libraries:
import pandas as pd
import numpy as np
import cv2
import os, sys
from PIL import Image
from imblearn.under_sampling import RandomUnderSampler

import keras.optimizers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, MaxPool2D, Dense, Dropout
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sn


# 1. accessing the data:
train_file_path_txt = '/Users/mohammedlajam/Documents/GitHub/COVID19-Detection/Datasets/test.txt'
train_path = '/Users/mohammedlajam/Desktop/test'

# 2. counting the number of images in the datset
count = 0
for path in os.listdir(train_path):
    # check if current path is a file
    if os.path.isfile(os.path.join(train_path, path)):
        count += 1
print('File count:', count)

### new:
train_df = pd.read_csv(train_file_path_txt)
# creating a column-names for each column for both train and test datasets
# dropping the 'patient id and 'data source' as they are not important in our case
train_df.columns = ['patient id', 'filename', 'class', 'data source']
train_df = train_df.drop(['patient id', 'data source'], axis=1)

# setting both the train and test dataset to same subjects (lowest number) to avoid biasing
# first, we need to classify the dataset (which are positive and which are negative)
x = train_df.drop(['class'], axis=1)
y = train_df['class']

# 1.2. Data Balancing:
print(y.value_counts())  # before resampling

rus = RandomUnderSampler(sampling_strategy=1)
x_res, y_res = rus.fit_resample(x, y)
train_df = x_res.join(y_res)  # joining the 2 columns in one table into a variable train_df
#train_df = shuffle(train_df)  # shuffling the dataset
print(train_df['class'].value_counts())  # after resampling
print(train_df.head())

train_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input, rescale=1.0/255.0, validation_split=0.2)
train_gen = train_datagen.flow_from_dataframe(dataframe=train_df, directory=train_path, x_col='filename',
                                              y_col='class', classes=['positive', 'negative'],
                                              target_size=(200, 200), batch_size=398, class_mode='binary')
x_train, y_train = next(train_gen)
print('shapes:')
print(x_train.shape)
print(y_train.shape)

# shape of the images:


# Scatter plot of the data:

# Historgram:

# Box and whisker plot:

