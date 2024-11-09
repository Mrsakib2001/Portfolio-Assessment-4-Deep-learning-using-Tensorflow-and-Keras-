import os
import random
import shutil

base_dir = "C:\\Users\\61449\\Downloads\\Portfolio_Assessment\\corrosion"  # Update if needed
test_dir = "C:\\Users\\61449\\Dwonloads\\Portfolio_Assessment\\test_images"


# Ensure test directories exist
os.makedirs(os.path.join(test_dir, 'rust'), exist_ok=True)
os.makedirs(os.path.join(test_dir, 'no_rust'), exist_ok=True)

# Select 10 random images from each class
for category in ['rust', 'no_rust']:
    images = os.listdir(os.path.join(base_dir, category))
    test_images = random.sample(images, 10)  # Randomly select 10 images
    for img in test_images:
        src = os.path.join(base_dir, category, img)
        dst = os.path.join(test_dir, category, img)
        shutil.move(src, dst)  # Move images to test set
