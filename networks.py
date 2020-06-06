import os, shutil, glob
import random
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LSTM
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import image
from enum import Enum
from datetime import datetime
from imutils import paths
import cv2
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

class LearningMode(Enum):
    CONVOLUTIONAL = 1
    SIMPLE_NN = 2

class ExecutionMode(Enum):
    LEARNING = 1
    TEST = 2

#class Config:
#    def __init__(self, mode=LearningMode.CONVOLUTIONAL, execution_mode = ExecutionMode.LEARNING):
#        self.execution_mode = execution_mode
#        self.mode = mode
#        self.base_data_dir = './data/preprocessed/Convolutional/'
#        self.train_dir = './data/train/'
#        self.val_dir = './data/val/'
#        self.test_dir = './data/test/'
#        self.model_path = os.path.join('models', mode.name + '.model')
#        #self.p_path = os.path.join('pickles', mode.name + '.p')
#        self.size = 100
#        self.accuracy_chart_path = os.path.join('charts', datetime.now().strftime("%d_%m_%Y_%H_%M_%S") + '.' + mode.name + '.model.jpg')
#        self.epochs = 10
#        self.batch_size = 100

class Config:
    def __init__(self, mode=LearningMode.SIMPLE_NN, execution_mode = ExecutionMode.LEARNING):
        self.execution_mode = execution_mode
        self.mode = mode
        self.base_data_dir = './data/preprocessed/SimpleNN/'
        self.train_dir = './data/train/'
        self.val_dir = './data/val/'
        self.test_dir = './data/test/'
        self.model_path = os.path.join('models', mode.name + '.model')
        #self.p_path = os.path.join('pickles', mode.name + '.p')
        self.size = 100
        self.accuracy_chart_path = os.path.join('charts', datetime.now().strftime("%d_%m_%Y_%H_%M_%S") + '.' + mode.name + '.model.jpg')
        self.epochs = 10
        self.batch_size = 100

_config = Config(mode=LearningMode.SIMPLE_NN , execution_mode = ExecutionMode.LEARNING)

def remove_files(dir):
    files = glob.glob(dir + '*.jpg')
    for f in files:
        os.remove(f)

def get_files(dir):
    all_files = []
    for (dirpath, dirnames, filenames) in os.walk(dir):
        all_files.extend(filenames)
    
    return all_files

def split_files():
    sick_files = get_files(_config.base_data_dir + 'sick/')
    healthy_files = get_files(_config.base_data_dir + 'healthy/')

    sick_train_set_size = int(len(sick_files)*0.8)
    sick_val_set_size = int(len(sick_files)*0.1)

    healthy_train_set_size = int(len(healthy_files)*0.8)
    healthy_val_set_size = int(len(healthy_files)*0.1)

    for i in range(len(sick_files)):
        dest = ''
        if i < sick_train_set_size:
            dest = _config.train_dir + 'sick/'
        elif i >= sick_train_set_size and i < sick_train_set_size + sick_val_set_size:
            dest = _config.val_dir + 'sick/'
        elif i >= sick_train_set_size + sick_val_set_size:
            dest = _config.test_dir + 'sick/'
    
        shutil.copyfile(_config.base_data_dir+ 'sick/' + sick_files[i], dest + sick_files[i])

    for i in range(len(healthy_files)):
        dest = ''
        if i < healthy_train_set_size:
            dest = _config.train_dir + 'healthy/'
        elif i >= healthy_train_set_size and i < healthy_train_set_size + healthy_val_set_size:
            dest = _config.val_dir + 'healthy/'
        elif i >= healthy_train_set_size + healthy_val_set_size:
            dest = _config.test_dir + 'healthy/'
    
        shutil.copyfile(_config.base_data_dir+ 'healthy/' + healthy_files[i], dest + healthy_files[i])


def create_convolutional_model():
    model = Sequential()
    model.add(Conv2D((16), (3,3), activation='relu', input_shape=(_config.size, _config.size, 3)))
    model.add(MaxPooling2D(2,2))
    model.add(Conv2D((32), (3,3), activation='relu'))
    model.add(MaxPooling2D(2,2))
    model.add(Conv2D((64), (3,3), activation='relu'))
    model.add(MaxPooling2D(2,2))
    model.add(Conv2D((128), (3,3), activation='relu'))
    model.add(MaxPooling2D(2,2))
    model.add(Conv2D((128), (3,3), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    return model

def create_simpleNN_model():
    model = Sequential()
    model.add(Flatten(input_shape=(100,100,3)))
    model.add(Dense(2500, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(2, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    return model

def load_test_data():
    sick_files = glob.glob(_config.test_dir+ 'sick/*.jpg')
    healthy_files = glob.glob(_config.test_dir+ 'healthy/*.jpg')
    length = len(sick_files) + len(healthy_files)

    x = np.zeros((length, _config.size, _config.size, 3), dtype='float')
    y = np.zeros(length, dtype='float')

    for i in range(len(sick_files)):
        y[i] = 1
        img = image.load_img(sick_files[i], target_size=(_config.size, _config.size))
        x[i] = image.img_to_array(img)

    for i in range(len(healthy_files)):
        img = image.load_img(healthy_files[i], target_size=(_config.size, _config.size))
        x[i + len(sick_files)] = image.img_to_array(img)

    return x, y

def perform_one_hot_encoding(items):
    #lb = LabelBinarizer()
    items = lb.fit_transform(items)
    items = to_categorical(items)
    return items

def print_accuracy_chart(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']

    epochs = range(1, len(acc) +1)
    plt.plot(epochs, acc, 'bo', label='acc')
    plt.plot(epochs, val_acc, 'b', label='val_acc')
    plt.legend()
    plt.savefig(_config.accuracy_chart_path)

def convolutional_learning():
    model = create_convolutional_model()

    train_datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.2, height_shift_range=0.2, zoom_range=0.2)
    test_datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.15, height_shift_range=0.15, zoom_range=0.2)

    train_generator = train_datagen.flow_from_directory(_config.train_dir, target_size=(_config.size, _config.size), batch_size=_config.batch_size, class_mode='binary')
    val_generator = test_datagen.flow_from_directory(_config.val_dir, target_size=(_config.size, _config.size), batch_size=_config.batch_size, class_mode='binary')

    cb = [ModelCheckpoint(_config.model_path, monitor='val_acc', save_best_only=True)]
    
    history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=_config.epochs , validation_data=val_generator, validation_steps=50, callbacks=cb)

    print_accuracy_chart(history)

def simpleNN_learning():
    print("[INFO] loading images...")
    imagePaths = list(paths.list_images(_config.train_dir))
    data = []
    labels = []

# loop over the image paths
    for imagePath in imagePaths:
        # extract the class label from the filename
        label = imagePath.split(os.path.sep)[-2]

        # load the image, swap color channels, and resize it to be a fixed
        
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (_config.size, _config.size))

        # update the data and labels lists, respectively    
        data.append(image)
        labels.append(label)


    model = create_simpleNN_model()

    data = np.array(data)
    labels = np.array(labels)

    labels = perform_one_hot_encoding(labels)

    cb = [ModelCheckpoint(_config.model_path, monitor='val_acc', save_best_only=True)]
    history = model.fit(data, labels, batch_size=_config.batch_size, epochs=_config.epochs, validation_split=0.2, callbacks=cb)
    print_accuracy_chart(history)

remove_files(_config.train_dir + 'sick/')
remove_files(_config.val_dir + 'sick/')
remove_files(_config.test_dir + 'sick/')
remove_files(_config.train_dir + 'healthy/')
remove_files(_config.val_dir + 'healthy/')
remove_files(_config.test_dir + 'healthy/')

split_files()

lb = LabelBinarizer()

if(_config.execution_mode == ExecutionMode.LEARNING):
    # Choose learning mode
    if(_config.mode == LearningMode.CONVOLUTIONAL):
        convolutional_learning()
    elif _config.mode == LearningMode.SIMPLE_NN:
        simpleNN_learning()


test_x, test_y = load_test_data()
model = load_model(_config.model_path)

if(_config.mode == LearningMode.SIMPLE_NN): #only SimpleNN model needs one hot encoding
    test_y = perform_one_hot_encoding(test_y)

result = model.evaluate(test_x, test_y, batch_size=64)
prediction = model.predict(test_x)

if(_config.mode == LearningMode.SIMPLE_NN): #only SimpleNN model needs one hot encoding
    test_y = np.argmax(test_y, axis=1)
    prediction = np.argmax(prediction, axis=1)

print(classification_report(test_y, prediction.round()))

cm = confusion_matrix(test_y, prediction.round())
total = sum(sum(cm))
acc = (cm[0, 0] + cm[1, 1]) / total
sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

 # show the confusion matrix, accuracy, sensitivity, and specificity
print(cm)
print("acc: {:.4f}".format(acc))
print("sensitivity: {:.4f}".format(sensitivity))
print("specificity: {:.4f}".format(specificity))


print(result)
print(prediction)
