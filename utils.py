# from tensorflow import lite
# from pycoral.utils import edgetpu
#
# def load_model(model_path):
#     interpreter = lite.Interpreter(model_path)
#     interpreter.allocate_tensors()
#     return interpreter
#
# def load_edge_model(model_path):
#     interpreter = edgetpu.make_interpreter(model_path)
#     interpreter.allocate_tensors()
#     return interpreter
INPUT_SIZE = (224, 224)
NUM_CLASSES = 16
BATCH_SIZE = 1
DATASET_FOLDER = '.'
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score


def get_test_data_generator():
    gen = ImageDataGenerator()
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
        interpreter.invoke()
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
