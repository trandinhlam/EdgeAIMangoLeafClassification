import time
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras import callbacks
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.efficientnet import EfficientNetB2
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Dense, GlobalAveragePooling2D, BatchNormalization, Input, Add, ZeroPadding2D, Flatten, Conv2D, AveragePooling2D, MaxPooling2D,MaxPool2D
from tensorflow.keras.optimizers import SGD
import gdown


# INPUT_SIZE = (500, 333)
INPUT_SIZE = (224, 224)
NUM_CLASSES = 16
BATCH_SIZE = 1
LEARNING_RATE = 0.01
EPOCHS = 50
MOMENTUM=0.1
MODEL_FOLDER = '/content/drive/MyDrive/Colab/Mango_pest_classification/version1/'
TEACHER_MODEL_NAME='ResNet50'

import os

def get_teacher_model(model):
  base_model = model(include_top=False, weights='imagenet')
  # add a global spatial average pooling layer
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  # let's add a fully-connected layer
  x = Dense(1024, activation='relu')(x)
  # and a logistic layer
  predictions = Dense(NUM_CLASSES, activation='softmax')(x)
  # this is the model we will train
  model = Model(inputs=base_model.input, outputs=predictions)
  # first: train only the top layers (which were randomly initialized)
  for layer in base_model.layers:
      layer.trainable = False
  return model

def load_model(model,weight_file):
  sgd = SGD(learning_rate=LEARNING_RATE, momentum=MOMENTUM)
  acc = CategoricalAccuracy(name='acc')
  model2 = get_teacher_model(model)
  model2.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=[acc])
  model2.load_weights(weight_file)
  return model2

gen = ImageDataGenerator()
test = gen.flow_from_directory('test/',
                               target_size=INPUT_SIZE,
                               batch_size=BATCH_SIZE,
                               shuffle=False)

def test_model(model):
  print('test:', model.evaluate(test))
  # Measure inference time during evaluation
  start_time = time.time()

  # Iterate over batches in the test generator
  i=0
  for batch_data, batch_labels in test:
    # Perform inference on the batch
    predictions = model.predict(batch_data)
    i=i+1
    if i>100:
       break


    # Calculate average inference time
  end_time = time.time()
  inference_time_per_batch = (end_time - start_time) / i
  print(f"Average Inference Time per Batch: {inference_time_per_batch} seconds")


from urllib.request import urlretrieve

def download_weights(url, destination):
    urlretrieve(url, destination)

def download_weights_from_drive(gdrive_file_id, destination):
    url = f'https://drive.google.com/uc?id={gdrive_file_id}'
    gdown.download(url, destination, quiet=False)

# Sử dụng hàm download_weights với URL cụ thể
models = [ResNet50,VGG16,VGG19,EfficientNetB2]
model_names = ['ResNet50','VGG16','VGG19','EfficientNetB2']
weight_ids = ['1-4y-rnhtwuPWX7jXaYSXKJs6RRXDdcOY',
          '1-RULAiXVz_s1gc-xltMiLVt9dz2Uv-rX',
          '1-6y-NTI-MfOdqw6F6FUkLoBkJPynPmkU',
          '1-4H87S4xQtA8gRv6FGWMc5DJMiWsOt-s']

for i in range(len(models)):
    model=models[i]
    model_name = model_names[i]
    weight_id=weight_ids[i]
    destination_path = f"{model_name}.hdf5"
    # if model_name !='ResNet50':
    download_weights_from_drive(weight_id, destination_path)
    print(f"load model {destination_path}")
    teacher = load_model(model, destination_path)
    test_model(teacher)