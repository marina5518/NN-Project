def preprocess_image(image, normalize=True):
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image)
    if normalize:
        image = image / 255.0   # Only for MobileNetV2
    image = np.expand_dims(image, axis=0)
    return image
