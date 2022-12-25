# Import các thư viện cần thiết & một số hằng số

from keras.utils.data_utils import get_file
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import EfficientNetB2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator

INPUT_SIZE = (224, 224)
NUM_CLASSES = 16
PATCH_SIZE = 64
MODEL_NAME = 'EfficientNetB2'
DATASET_FOLDER = '.'
LEARNING_RATE = 0.01
EPOCHS = 50
MOMENTUM = 0.1
MODEL_LINK = 'https://drive.google.com/file/d/1-omJp4YaAL4cczyOIiPzPVVtV2Qxs-UF/view?usp=share_link'

base_model = EfficientNetB2(include_top=False, weights='imagenet')

# Load dữ liệu
gen = ImageDataGenerator()

test = gen.flow_from_directory(DATASET_FOLDER + '/test/',
                               target_size=INPUT_SIZE,
                               batch_size=PATCH_SIZE,
                               shuffle=False)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Xây dựng model dựa trên các backbone thông dụng (VGG,Resnet,Xception,Inception...)
def getModel():
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


def getCustomModel():
    # add a global spatial average pooling layer
    x = base_model.output
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
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


def singleTest():
    model2 = getModel()
    model2.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=[acc])
    model2.load_weights('model.hdf5')
    print(f'test:', model2.evaluate(test))


def print_hi(name):
    singleTest()


if __name__ == '__main__':
    print_hi('PyCharm')
