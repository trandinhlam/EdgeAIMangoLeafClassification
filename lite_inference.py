import numpy as np
from tensorflow import lite
from sklearn.metrics import accuracy_score

INPUT_SIZE = (224, 224)
NUM_CLASSES = 16
BATCH_SIZE = 1
DATASET_FOLDER = '.'
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import argparse

# Load dữ liệu
gen = ImageDataGenerator()

test = gen.flow_from_directory(DATASET_FOLDER + '/test/',
                               target_size=INPUT_SIZE,
                               batch_size=BATCH_SIZE,
                               shuffle=False)


def load_model(model_path):
    interpreter = lite.Interpreter(model_path)
    interpreter.allocate_tensors()
    return interpreter


def get_io_index(interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return input_details[0]['index'], output_details[0]['index']


def get_model_path():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model', required=True, help='File path of .tflite file.')
    args = parser.parse_args()
    return args.model


if __name__ == '__main__':
    # interpreter = load_model('model_lite/model_full_integer_ResNet50_20230205-10:12:18.tflite')
    interpreter = load_model('model_lite/model_full_integer_ResNet50_20230205-10:12:18.tflite')
    # interpreter = load_model(get_model_path())
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_index, output_index = get_io_index(interpreter)
    Y_predict = []
    Y_test = []
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
    print(acc)
