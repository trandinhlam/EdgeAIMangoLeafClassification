from PIL import Image

from pycoral.adapters import common
from pycoral.utils import edgetpu


def load_model(model_path):
    interpreter = edgetpu.make_interpreter(model_path)
    interpreter.allocate_tensors()
    return interpreter


def do_inference(model_path, test_img):
    interpreter = load_model(model_path)
    size = common.input_size(interpreter)
    image = Image.open(test_img).convert('RGB').resize(size, Image.ANTIALIAS)
    common.set_input(interpreter, image)
    result = interpreter.invoke()
    print(result)


if __name__ == '__main__':
    do_inference('./model_edge_tpu/model_ResNet50_20230203-21:48:25_edgetpu.tflite', './test/normal/normal30.jpg')
