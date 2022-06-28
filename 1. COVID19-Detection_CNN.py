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
import numpy as np
from sklearn.utils import resample

# 1. Data Preprocessing:
# 1.1. preparation of text table using pandas:
# first, in the txt files, columns are separated by SPACE. Hence, we need to change all SPACES to
# ',' in order to be identified as columns in pandas.
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

print(train_df.head())  # see the first 5 rows and columns of train
print(train_df['class'].value_counts())  # to know the number of positives and negatives.
print(test_df['class'].value_counts())

# setting both the train and test dataset to same subjects (lowest number) to avoid biasing
# first, we need to classify the dataset (which are positive and which are negative)
positive = train_df[train_df['class'] == 'positive']
negative = train_df[train_df['class'] == 'negative']

df_majority_downsampled = resample(positive, replace=True, n_samples=13991)
train_df = pd.concat([negative, df_majority_downsampled])

print(train_df['class'].value_counts())
#test