import tensorflow as tf
import numpy as np
import os
import random
from PIL import Image

model_input_width = 640
model_input_height = 640
resize_side = int(min(model_input_width, model_input_height) * 1.12)
calib_images_path = 'datasets/coco/images/train2017'


# Preprocessing function (resize and crop)
def preproc(image, output_height=640, output_width=640, resize_side=resize_side):
    h, w = image.shape[0], image.shape[1]
    scale = tf.cond(tf.less(h, w), lambda: resize_side / h, lambda: resize_side / w)
    resized_image = tf.image.resize(tf.expand_dims(image, 0), [int(h * scale), int(w * scale)])
    cropped_image = tf.image.resize_with_crop_or_pad(resized_image, output_height, output_width)
    return tf.squeeze(cropped_image)

# Apply preprocessing to all images in the calibration folder
images_list = random.sample([img for img in os.listdir(calib_images_path) if img.endswith(".jpg")], 1024)
calib_dataset = np.zeros((len(images_list), model_input_height, model_input_width, 3))

for idx, img_name in enumerate(sorted(images_list)):
    img = np.array(Image.open(os.path.join(calib_images_path, img_name)).convert('RGB'))
    img_preproc = preproc(img)
    calib_dataset[idx] = img_preproc.numpy()

np.save("calib_set.npy", calib_dataset)  # Save for reuse