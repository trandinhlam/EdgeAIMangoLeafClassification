import argparse

from PIL import Image,ImageDraw, ImageFont

from tensorflow import lite
import utils
import numpy as np
import os

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
    image = Image.fromarray(test_img).resize(size, Image.BILINEAR)
    interpreter.tensor(input_index)()[0][:, :] = image
    interpreter.invoke()
    predict = interpreter.get_tensor(output_index)
    predict = np.argmax(predict, axis=1)
    classes = utils.get_test_data_generator().class_indices
    class_id = predict[0]
    label = {i for i in classes if classes[i] == class_id}
    print(predict, label)
    return class_id, list(label)[0]

from flask import Flask, jsonify, request
import base64
from PIL import Image
from io import BytesIO
import os
from flask_cors import CORS  # Import the CORS extension

app = Flask(__name__)
CORS(app)

interpreter = load_interpreter('model_lite/model_full_integer_ResNet18_20230723-10_57_21.tflite')
folder_path = 'img_inference' 

# Get a list of all files in the folder
files = os.listdir(folder_path)
image_files = [file for file in files if file.lower().endswith(('.jpeg', '.jpg'))]

@app.route('/api/data', methods=['GET'])
def load_data():
    resp = []
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        img = Image.open(image_path)
        predict, label = do_inference(interpreter, np.uint8(img))
        print(predict, label, image_file)
        # Convert image data to base64
        buffered = BytesIO()
        img = img.resize((500, 500))
        img.save(buffered, format="PNG")  # Change the format as needed
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8") 
        resp.append({"label": label, "img": img_base64})
    return jsonify(resp)



if __name__ == '__main__':
    app.run(debug=True)
    # interpreter = load_model('model_lite/model_full_integer_ResNet50_20230205-10:12:18.tflite')
    # interpreter = load_interpreter('model_lite/model_full_integer_ResNet8_20230219-08:20:54.tflite')
    # interpreter = load_model(get_model_path())
    # test_gen = utils.get_test_data_generator()
    # acc = utils.calculate_acc(interpreter, test_gen)
    # print(acc)
    # Filter the list to include only files with certain extensions (e.g., '.jpeg' or '.jpg')

    # Iterate through each image file and open it
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        img = Image.open(image_path)
        # img = np.uint8(img)
        predict, label = do_inference(interpreter, np.uint8(img))
        print(predict, label, image_file)
        if label is not None:
            
            # img = cv2.putText(img=img,
            #                         text=f'label: {label}',
            #                         org=(0, 75), fontFace=cv2.FONT_HERSHEY_TRIPLEX,
            #                         fontScale=0.75,
            #                         color=(255, 0, 0), thickness=2)
        # img = Image.fromarray(img)
            draw = ImageDraw.Draw(img)
            font_path = ImageFont.load_default().font_variant(size=20)
            # font = ImageFont.truetype("arial.ttf", 20)  # You can adjust the font and size
            draw.text((0, 75), f"label: {label}", font=font_path, fill=(255, 0, 0))
            img.save(os.path.join('img_inference_result', image_file))
