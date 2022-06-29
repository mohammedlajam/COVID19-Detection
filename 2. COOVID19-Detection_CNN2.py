'''
Dataset: there are 2 files for train and test dataset (Images). these files have only
image file names and it does not show the class (Positive or Negative) of the image.
additionally, there are 2 extra txt files for train and test sets showing the Patient ID, filename,
class (Positive or Negative) and data source. By using those files, we can link which images have
positive and which have negative (for both train and test datasets)

Important: the dataset can be downloaded from the following link:
https://www.kaggle.com/datasets/andyczhao/covidx-cxr2
'''

# Libraries:
import pandas as pd
import tensorflow as tf
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils import shuffle
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, MaxPool2D, Dense, Dropout
import os
import PIL
import cv2

import tensorflow as tf
from tensorflow import keras


# 1. Data Preprocessing:
# 1.1. preparation of text table using pandas:
'''first, in the txt files, columns are separated by SPACE. Hence, we need to change all SPACES to
',' in order to be identified as columns in pandas.'''
train_file_path_txt = '/Users/mohammedlajam/Documents/GitHub/COVID19-Detection/Datasets/train.txt'
test_file_path_txt = '/Users/mohammedlajam/Documents/GitHub/COVID19-Detection/Datasets/test.txt'
train_path = '/Users/mohammedlajam/Documents/GitHub/COVID19-Detection/Datasets/train'
test_path = '/Users/mohammedlajam/Documents/GitHub/COVID19-Detection/Datasets/test'

train_df = pd.read_csv(train_file_path_txt)
test_df = pd.read_csv(test_file_path_txt)
# creating a column-names for each column for both train and test datasets
# dropping the 'patient id and 'data source' as they are not important in our case

train_df.columns = ['patient id', 'filename', 'class', 'data source']
train_df = train_df.drop(['patient id', 'data source'], axis=1)
test_df.columns = ['patient id', 'filename', 'class', 'data source']
test_df = test_df.drop(['patient id', 'data source'], axis=1)

# setting both the train and test dataset to same subjects (lowest number) to avoid biasing
# first, we need to classify the dataset (which are positive and which are negative)
x = train_df.drop(['class'], axis=1)
y = train_df['class']
print(y.value_counts())  # before resampling

rus = RandomUnderSampler(sampling_strategy=1)
x_res, y_res = rus.fit_resample(x, y)
train_df = x_res.join(y_res)  # joining the 2 columns in one table into a variable train_df
train_df = shuffle(train_df)  # shuffling the dataset

print(train_df['class'].value_counts())  # after resampling
print(train_df.head())

# using flow_from_dataframe function to link the class of the txt file to the image folder
train_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input, rescale=1.0/255.0, validation_split=0.2)
test_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input, rescale=1.0/255.0, validation_split=0.2)

# preparing the data in a format that the model expects using flow_from_dataframe:
train_gen = train_datagen.flow_from_dataframe(dataframe=train_df, directory=train_path, x_col='filename',
                                              y_col='class', classes=['positive', 'negative'],
                                              target_size=(200, 200), batch_size=64, class_mode='binary')
test_gen = test_datagen.flow_from_dataframe(dataframe=test_df, directory=test_path, x_col='filename',
                                            y_col='class', classes=['positive', 'negative'],
                                            target_size=(200, 200), batch_size=64, class_mode='binary', shuffle=False)

x_train, y_train = next(train_gen)
x_test, y_test = next(test_gen)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# plotting an image as an example:
plt.imshow(x_train[10])
plt.show()
print(y_train[10])
'''
def Plot_images(self, imges_arr):
    fig, axes = plt.subplot(1, 10, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(imges_arr, axes):
        ax.imshow(img)
        ax.axes('off')
    plt.tight_layout()
    plt.show()
'''

