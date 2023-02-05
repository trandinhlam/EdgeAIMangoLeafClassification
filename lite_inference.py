import argparse

from PIL import Image
from tensorflow import lite

import utils
import numpy as np


def get_model_path():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--model', required=True, help='File path of .tflite file.')
    args = parser.parse_args()
    return args.model


def load_interpreter(model_path):
    interpreter = lite.Interpreter(model_path)
    interpreter.allocate_tensors()
    return interpreter


def get_input_size(interpreter):
    _, height, width, _ = interpreter.get_input_details()[0]('shape')
    return width, height


def do_inference(interpreter, test_img):
    size = utils.INPUT_SIZE
    input_index, output_index = utils.get_io_index(interpreter)
    image = Image.fromarray(test_img).resize(size, Image.ANTIALIAS)
    interpreter.tensor(input_index)()[0][:, :] = image
    interpreter.invoke()
    predict = interpreter.get_tensor(output_index)
    predict = np.argmax(predict, axis=1)
    return predict


if __name__ == '__main__':
    # interpreter = load_model('model_lite/model_full_integer_ResNet50_20230205-10:12:18.tflite')
    interpreter = load_interpreter('model_lite/model_full_integer_ResNet50_20230205-10:52:35.tflite')
    # interpreter = load_model(get_model_path())
    test_gen = utils.get_test_data_generator()
    acc = utils.calculate_acc(interpreter, test_gen)
    print(acc)
