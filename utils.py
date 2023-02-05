INPUT_SIZE = (224, 224)
NUM_CLASSES = 16
BATCH_SIZE = 1
DATASET_FOLDER = '.'
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
import time

gen = ImageDataGenerator()


def get_valid_data_generator():
    valid = gen.flow_from_directory(DATASET_FOLDER + '/valid/',
                                    target_size=INPUT_SIZE,
                                    batch_size=BATCH_SIZE,
                                    shuffle=False)
    return valid


def get_test_data_generator():
    test = gen.flow_from_directory(DATASET_FOLDER + '/test/',
                                   target_size=INPUT_SIZE,
                                   batch_size=BATCH_SIZE,
                                   shuffle=False)
    return test


def get_io_index(interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return input_details[0]['index'], output_details[0]['index']


def calculate_acc(interpreter, test_generator):
    input_index, output_index = get_io_index(interpreter)
    Y_predict = []
    Y_test = []
    test = test_generator
    for i in range(len(test)):
        x_test, y_test = next(test)
        x_test = x_test.astype(np.uint8)
        interpreter.set_tensor(input_index, x_test)
        start = time.perf_counter()
        interpreter.invoke()
        inference_time = time.perf_counter() - start
        print('%.1fms' % (inference_time * 1000))
        predict = interpreter.get_tensor(output_index)
        predict = np.argmax(predict, axis=1)
        print('predict:', predict)
        Y_predict.append(predict)
        y_test = np.argmax(y_test, axis=1)
        print('y_test:', y_test)
        Y_test.append(y_test)
    print(len(Y_predict))
    acc = accuracy_score(Y_test, Y_predict, normalize=True)
    return acc
