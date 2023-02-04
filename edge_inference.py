from PIL import Image

from pycoral.adapters import common, classify
from pycoral.utils import edgetpu
INPUT_SIZE = (224, 224)
NUM_CLASSES = 16
BATCH_SIZE = 1
DATASET_FOLDER = '.'
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Load dữ liệu
gen = ImageDataGenerator()

test = gen.flow_from_directory(DATASET_FOLDER + '/test/',
                               target_size=INPUT_SIZE,
                               batch_size=BATCH_SIZE,
                               shuffle=False)

def load_model(model_path):
    interpreter = edgetpu.make_interpreter(model_path)
    interpreter.allocate_tensors()
    return interpreter


def do_inference(model_path, test_img):
    interpreter = load_model(model_path)
    size = common.input_size(interpreter)
    image = Image.open(test_img).convert('RGB').resize(size, Image.ANTIALIAS)
    common.set_input(interpreter, image)
    interpreter.invoke()
    classes = classify.get_classes(interpreter, top_k=1)
    print(classes)


def get_io_index(interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return input_details[0]['index'], output_details[0]['index']


if __name__ == '__main__':
    # do_inference('./model_edge_tpu/model_ResNet50_20230203-21:48:25_edgetpu.tflite', './test/normal/normal30.jpg')
    interpreter = load_model('./model_edge_tpu/model_ResNet50_20230203-21:48:25_edgetpu.tflite')
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_index, output_index = get_io_index(interpreter)
    y_predict = []
    for i in range(len(test)):
        x_test, y_test = next(test)
        x_test = x_test.astype(np.uint8)
        interpreter.set_tensor(input_index, x_test)
        interpreter.invoke()
        predict = interpreter.get_tensor(output_index)
        print('predict:', predict)
        y_predict.append(predict)
        print('y_test:', y_test)
    print(len(y_predict))
