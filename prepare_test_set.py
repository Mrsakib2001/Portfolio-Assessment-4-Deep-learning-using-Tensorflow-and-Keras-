import os
import random
import shutil

# Define paths
base_dir = "C:\\Users\\61449\\PycharmProjects\\Portfolio_Assessment\\log_labelled"
test_dir = "C:\\Users\\61449\\PycharmProjects\\Portfolio_Assessment\\log_test_images"

# Ensure test directories exist
os.makedirs(test_dir, exist_ok=True)

# Select 10 random images for testing
images = [img for img in os.listdir(base_dir) if img.endswith(('.jpg', '.png'))]
test_images = random.sample(images, 10)  # Randomly select 10 images

for img in test_images:
    src_img = os.path.join(base_dir, img)
    src_json = os.path.join(base_dir, img.replace('.jpg', '.json').replace('.png', '.json'))
    dst_img = os.path.join(test_dir, img)
    dst_json = os.path.join(test_dir, img.replace('.jpg', '.json').replace('.png', '.json'))

    # Move both image and its corresponding annotation JSON
    shutil.move(src_img, dst_img)
    if os.path.exists(src_json):
        shutil.move(src_json, dst_json)

print("Test images and corresponding JSON files moved successfully.")
