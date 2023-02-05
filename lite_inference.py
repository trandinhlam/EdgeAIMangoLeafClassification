import numpy as np
from tensorflow import lite
from sklearn.metrics import accuracy_score

import argparse
import utils


def get_model_path():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model', required=True, help='File path of .tflite file.')
    args = parser.parse_args()
    return args.model


def load_model(model_path):
    interpreter = lite.Interpreter(model_path)
    interpreter.allocate_tensors()
    return interpreter


if __name__ == '__main__':
    # interpreter = load_model('model_lite/model_full_integer_ResNet50_20230205-10:12:18.tflite')
    interpreter = load_model('model_lite/model_full_integer_ResNet50_20230205-10:12:18.tflite')
    # interpreter = load_model(get_model_path())
    test_gen = utils.get_test_data_generator()
    acc = utils.calculate_acc(interpreter, test_gen)
    print(acc)
