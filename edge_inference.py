from PIL import Image
from pycoral.adapters import common, classify
from pycoral.utils import edgetpu

import utils


def load_interpreter(model_path):
    interpreter = edgetpu.make_interpreter(model_path)
    interpreter.allocate_tensors()
    return interpreter


def do_inference(model_path, test_img):
    interpreter = load_interpreter(model_path)
    size = common.input_size(interpreter)
    image = Image.open(test_img).convert('RGB').resize(size, Image.ANTIALIAS)
    common.set_input(interpreter, image)
    interpreter.invoke()
    classes = classify.get_classes(interpreter, top_k=1)
    print(classes)


if __name__ == '__main__':
    # do_inference('./model_edge_tpu/model_ResNet50_20230203-21:48:25_edgetpu.tflite', './test/normal/normal30.jpg')
    interpreter = load_interpreter('./model_edge_tpu/model_full_integer_ResNet8_20230218-11:26:04_edgetpu.tflite')
    test_gen = utils.get_test_data_generator()
    acc = utils.calculate_acc(interpreter, test_gen)
    print(acc)
