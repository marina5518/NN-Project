from PIL import Image
import numpy as np

IMG_SIZE = 224

def preprocess_image(image, model_name="CNN"):
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image, dtype=np.float32)

    if model_name == "MobileNetV2":
        image = image / 255.0  # MobileNetV2 needs [0, 1] — no internal rescaling

    # CNN has Rescaling(1/255) built-in, so it needs raw [0, 255]

    image = np.expand_dims(image, axis=0)
    return image
