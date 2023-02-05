import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

INPUT_SIZE = (224, 224)
NUM_CLASSES = 16
BATCH_SIZE = 1
DATASET_FOLDER = '..'
LEARNING_RATE = 0.01
EPOCHS = 50
MOMENTUM = 0.1

IMG_SHAPE = (224, 224, 3)

gen = ImageDataGenerator()

valid = gen.flow_from_directory(DATASET_FOLDER + '/valid/',
                                target_size=INPUT_SIZE,
                                batch_size=BATCH_SIZE,
                                shuffle=True)


def dataset():
    for _ in range(100):
        data = np.random.rand(1, 244, 244, 3)
        yield [data.astype(np.float32)]


def representative_dataset():
    for x, _ in valid:
        print(x.shape)
        yield [x]


if __name__ == '__main__':
    print(dataset())
    print(representative_dataset())
