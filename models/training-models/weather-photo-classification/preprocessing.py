from PIL import Image
import os
import numpy as np

# Function to load images from a directory
def load_images_from_dir(directory, width, height):
    images = []
    labels = []
    for label in os.listdir(directory):
        label_path = os.path.join(directory, label)
        for filename in os.listdir(label_path):
            img_path = os.path.join(label_path, filename)
            img = Image.open(img_path)
            img = img.resize((width, height))
            images.append(np.array(img))
            labels.append(label)
    return np.array(images), np.array(labels)

# Example usage
directory = "path_to_your_image_directory"
X, y = load_images_from_dir(directory)