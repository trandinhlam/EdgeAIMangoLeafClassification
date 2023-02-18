from datetime import datetime

import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
import numpy as np

import resnet8
import utils
from resnet8 import ResNet8

dateStr = datetime.now().strftime("%Y%m%d-%H:%M:%S")

INPUT_SIZE = (224, 224)
NUM_CLASSES = 16
BATCH_SIZE = 1
DATASET_FOLDER = '.'
LEARNING_RATE = 0.01
EPOCHS = 50
MOMENTUM = 0.1

IMG_SHAPE = (224, 224, 3)
inputs = Input(shape=IMG_SHAPE, batch_size=BATCH_SIZE)
MODEL_NAME = 'ResNet8'
base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=inputs)


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


def representative_dataset():
    valid = utils.get_valid_data_generator()
    for i in range(len(valid)):
        x_valid, _ = next(valid)
        i += 1
        print(i, x_valid.shape)
        yield [x_valid]


def convert_to_lite(model_file):
    converter = lite.TFLiteConverter.from_keras_model(model_file)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.SELECT_TF_OPS]
    converter.representative_dataset = representative_dataset
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    tfmodel = converter.convert()
    open(f"model_lite/model_full_integer_{MODEL_NAME}_{dateStr}.tflite", "wb").write(tfmodel)


def build_model():
    model = get_model()
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=[acc])
    model.load_weights(f'model_h5/model_{MODEL_NAME}.hdf5')
    model.summary()
    return model


def single_test(model):
    test = utils.get_test_data_generator()
    print(f'test:', model.evaluate(test))


if __name__ == '__main__':
    # convert_to_lite(build_model())
    # single_test(build_model())

    student = resnet8.resnet8_sequential()
    sgd = SGD(learning_rate=LEARNING_RATE, momentum=MOMENTUM)
    acc = CategoricalAccuracy(name='acc')
    student.build(input_shape=(1, 224, 224, 3))
    student.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=[acc])
    student.load_weights('./model_h5/ResNet8/student_base_best_230218.hdf5')
    # student.load_weights('./model_h5/ResNet8/ResNet8_best_1676130933.1550047_01.tf')
    # student.load_weights('./model_h5/Resnet8_230217/check_point')
    student.summary()
    single_test(student)
    convert_to_lite(student)

