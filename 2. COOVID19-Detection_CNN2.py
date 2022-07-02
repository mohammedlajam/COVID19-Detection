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
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, MaxPool2D, Dense
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sn


# 1. Data Preprocessing:
# 1.1. preparation of text table using pandas:
'''first, in the txt files, columns are separated by SPACE. Hence, we need before reading the 
files to change all SPACEs to COMMAs in order to be identified as columns in pandas.'''
# reading the txt files and the folders of images from the local directory:
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

# 1.2. Data Balancing:
print(y.value_counts())  # before resampling
rus = RandomUnderSampler(sampling_strategy=1)
x_res, y_res = rus.fit_resample(x, y)
train_df = x_res.join(y_res)  # joining the 2 columns in one table into a variable train_df
train_df = shuffle(train_df)  # shuffling the dataset
print(train_df['class'].value_counts())  # after resampling
print(train_df.head())

# 1.3. Splitting the training set into train and validation data:
train_df, valid_df = train_test_split(train_df, train_size=0.9, random_state=0)
print('train, validation and test sets:')
print(f"Negative and positive values of train: {train_df['class'].value_counts()}")
print(f"Negative and positive values of validation: {valid_df['class'].value_counts()}")
print(f"Negative and positive values of test: {test_df['class'].value_counts()}")

# 1.4. link the class of the txt file to the image filename:
train_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input, rescale=1.0/255.0, validation_split=0.2)
test_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input, rescale=1.0/255.0, validation_split=0.2)

# preparing the data in a format that the model expects using flow_from_dataframe:
train_gen = train_datagen.flow_from_dataframe(dataframe=train_df, directory=train_path, x_col='filename',
                                              y_col='class', classes=['positive', 'negative'],
                                              target_size=(200, 200), batch_size=100, class_mode='binary')
valid_gen = test_datagen.flow_from_dataframe(dataframe=valid_df, directory=train_path, x_col='filename',
                                             y_col='class', classes=['positive', 'negative'],
                                             target_size=(200, 200), batch_size=100, class_mode='binary')
test_gen = test_datagen.flow_from_dataframe(dataframe=test_df, directory=test_path, x_col='filename',
                                            y_col='class', classes=['positive', 'negative'],
                                            target_size=(200, 200), batch_size=100, class_mode='binary', shuffle=False)

x_train, y_train = next(train_gen)
x_valid, y_valid = next(valid_gen)
x_test, y_test = next(test_gen)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# plotting an image as an example:
plt.imshow(x_train[10])  # plotting one image from the train set
plt.show()
print(y_train[10])
plt.imshow(x_test[10])  # plotting one image from the test set
plt.show()
print(y_test[10])

# 2. Building a Model:
# 2.1. Building a CNN Architechture:
model = Sequential([Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same',
                    input_shape=(200, 200, 3)),
                    MaxPool2D(pool_size=(2, 2), strides=2),
                    Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
                    MaxPool2D(pool_size=(2, 2), strides=2),
                    Flatten(),
                    Dense(units=2, activation='softmax'),
                    ])
model.summary()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 2.2. Training the Model:
# In this case during fitting the model, the x parameter only to be mentioned (y is not included)
# because the train_gen is generated from a function ImageDataGenerator, in which the labels
# are integrated along side with the images.
history = model.fit(x=train_gen, validation_data=valid_gen, epochs=10, steps_per_epoch=250, verbose=2)

# 2.3. Visualization of Accuracy vs Epochs:
plt.plot(history.history['accuracy'], label="Training accuracy")
plt.plot(history.history['val_accuracy'], label="Validation accuracy")
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.grid()
plt.legend()
plt.show()

# 3. Model Predictions:
predictions = model.predict(x=test_gen, verbose=0)
print(np.round(predictions))  # checking the shape of the output predictions

# 4. Model Evaluation:
print("Model Evaluation")
# 4.1. test accuracy and test loss:
test_loss, test_acc = model.evaluate(test_gen)
print(f'Model ACCURACY is {test_acc}')
print(f'Model LOSS is {test_loss}')

# 4.2. Classification report (Accuracy, Precision, Recall, F1-score)
print(classification_report(y_true=test_gen.classes, y_pred=np.argmax(predictions, axis=1)))

# 4.3. Confusion Matrix:
cm = confusion_matrix(y_true=test_gen.classes, y_pred=np.argmax(predictions, axis=1))
print(cm)
print(test_gen.class_indices)  # to show the corresponding classes to positive and negative
# plotting the confusion matrix using seaborn function:
plt.figure(figsize=(10, 7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()