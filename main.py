# Import các thư viện cần thiết & một số hằng số

import tensorflow as tf
from tensorflow.keras.applications.efficientnet import EfficientNetB2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Input
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from datetime import datetime
import numpy as np

dateStr = datetime.now().strftime("%Y%m%d-%H:%M:%S")

INPUT_SIZE = (224, 224)
NUM_CLASSES = 16
BATCH_SIZE = 1
MODEL_NAME = 'MobileNet'
DATASET_FOLDER = '.'
LEARNING_RATE = 0.01
EPOCHS = 50
MOMENTUM = 0.1
MODEL_LINK = 'https://drive.google.com/file/d/1-omJp4YaAL4cczyOIiPzPVVtV2Qxs-UF/view?usp=share_link'

IMG_SHAPE = (224, 224, 3)
# inputs = Input(shape=IMG_SHAPE, batch_size=BATCH_SIZE, dtype=np.uint8)
inputs = Input(shape=IMG_SHAPE, batch_size=BATCH_SIZE)
base_model = MobileNet(include_top=False, weights='imagenet', input_tensor=inputs)

# Load dữ liệu
gen = ImageDataGenerator()

test = gen.flow_from_directory(DATASET_FOLDER + '/test/',
                               target_size=INPUT_SIZE,
                               batch_size=BATCH_SIZE,
                               shuffle=False)


def print_devices():
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    # detect the TPU
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(resolver)
    # This is the TPU initialization code that has to be at the beginning.
    tf.tpu.experimental.initialize_tpu_system(resolver)
    print("All devices: ", tf.config.list_logical_devices('TPU'))


# Xây dựng model dựa trên các backbone thông dụng (VGG,Resnet,Xception,Inception...)
def get_model():
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


sgd = SGD(learning_rate=LEARNING_RATE, momentum=MOMENTUM)
acc = CategoricalAccuracy(name='acc')

from tensorflow import lite
import numpy as np


def representative_dataset():
    # testing purpose only
    for _ in range(100):
        data = np.random.rand(1, 244, 244, 3)
        yield [data.astype(np.float16)]


def convert_to_lite(model_file):
    converter = lite.TFLiteConverter.from_keras_model(model_file)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.representative_dataset = representative_dataset
    # converter.inference_input_type = tf.uint8
    # converter.inference_output_type = tf.uint8
    tfmodel = converter.convert()
    open(f"model_float16_{MODEL_NAME}_{dateStr}.tflite", "wb").write(tfmodel)


def build_model():
    model = get_model()
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=[acc])
    model.load_weights(f'model_{MODEL_NAME}.hdf5')
    model.summary()
    return modelg


def single_test():
    model = build_model()
    print(f'test:', model.evaluate(test))


if __name__ == '__main__':
    # printDevices()
    convert_to_lite(build_model())
    # single_test()
