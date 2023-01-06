# Importing Image class from PIL module
from PIL import Image

def merge(image1, image2):
    image1_size = image1.size
    image2_size = image2.size
    new_image = Image.new('RGB', (image1_size[0]+image2_size[0], image1_size[1]), (250, 250, 250))
    new_image.paste(image1, (0, 0))
    new_image.paste(image2, (image1_size[0], 0))
    return new_image

first_img=None
for i in range(41, 53):
    # Opens a image in RGB mode
    im = Image.open(f'/home/lamtd/Desktop/my2022/w{i}.jpg')
    width, height = im.size
    # Setting the points for cropped image
    left = 102
    top = 232
    right = left+615
    bottom = top+1004
    im1 = im.crop((left, top, right, bottom))
    if first_img is None:
        first_img = im1
    else:
        first_img = merge(first_img, im1)
first_img.show()

