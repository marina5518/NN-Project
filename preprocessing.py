from PIL import Image
import numpy as np

IMG_SIZE = 224

def preprocess_image(image):

    image = image.convert("RGB")

    image = image.resize((IMG_SIZE, IMG_SIZE))

    image = np.array(image)

    image = np.expand_dims(image, axis=0)

    return image
