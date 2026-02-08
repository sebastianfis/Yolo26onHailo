import tensorflow as tf
import numpy as np
import os
import json
import random
from PIL import Image
from collections import defaultdict

# ---------------- CONFIG ----------------

model_input_width = 640
model_input_height = 640
resize_side = int(min(model_input_width, model_input_height) * 1.12)

images_path = "datasets/coco/images/train2017"
annotations_path = "datasets/coco/annotations/instances_train2017.json"

TARGET_SIZE = 1024

# ---------------- PREPROCESS ----------------

def preproc(image,
            output_height=640,
            output_width=640,
            resize_side=resize_side):

    h, w = image.shape[0], image.shape[1]
    scale = resize_side / min(h, w)

    resized = tf.image.resize(
        tf.expand_dims(image, 0),
        [int(h * scale), int(w * scale)]
    )

    cropped = tf.image.resize_with_crop_or_pad(
        resized,
        output_height,
        output_width
    )

    return tf.squeeze(cropped)

# ---------------- LOAD COCO ----------------


with open(annotations_path, "r") as f:
    coco = json.load(f)

images_info = {img["id"]: img["file_name"]
               for img in coco["images"]}

# Per-image class mapping
img_classes = defaultdict(list)

for ann in coco["annotations"]:
    img_classes[ann["image_id"]].append(ann["category_id"])

# Unique classes per image
img_classes = {
    img_id: list(set(cats))
    for img_id, cats in img_classes.items()
}

# All classes
all_classes = sorted({
    ann["category_id"]
    for ann in coco["annotations"]
})

NUM_CLASSES = len(all_classes)
quota_per_class = TARGET_SIZE // NUM_CLASSES

print(f"{NUM_CLASSES} classes → "
      f"target {quota_per_class} images/class")

# ---------------- BUILD CLASS POOLS ----------------

class_to_images = defaultdict(list)

for img_id, cats in img_classes.items():
    for c in cats:
        class_to_images[c].append(img_id)

# Shuffle for randomness
for c in class_to_images:
    random.shuffle(class_to_images[c])

# ---------------- GREEDY SELECTION ----------------

selected_images = set()
class_counts = defaultdict(int)

# Pass 1 — fill quotas
for c in all_classes:

    for img_id in class_to_images[c]:

        if class_counts[c] >= quota_per_class:
            break

        if img_id in selected_images:
            continue

        selected_images.add(img_id)

        # Update all classes in that image
        for cc in img_classes[img_id]:
            class_counts[cc] += 1

# Pass 2 — fill remaining slots
remaining = TARGET_SIZE - len(selected_images)

if remaining > 0:

    # Prefer high class diversity
    diversity_sorted = sorted(
        img_classes.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )

    for img_id, _ in diversity_sorted:

        if len(selected_images) >= TARGET_SIZE:
            break

        if img_id in selected_images:
            continue

        selected_images.add(img_id)

print(f"Selected {len(selected_images)} images")

# ---------------- BUILD DATASET ----------------

selected_files = [
    images_info[img_id]
    for img_id in selected_images
]

calib_dataset = np.zeros(
    (len(selected_files),
     model_input_height,
     model_input_width,
     3),
    dtype=np.float32
)

for idx, file_name in enumerate(sorted(selected_files)):

    img = np.array(
        Image.open(os.path.join(images_path, file_name))
        .convert("RGB")
    )

    calib_dataset[idx] = preproc(img).numpy()

np.save("calib_set.npy", calib_dataset)

print("Saved class-balanced calib_set.npy")
