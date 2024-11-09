import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from mrcnn.config import Config
from mrcnn import model as modellib, utils
import skimage.draw

# Install missing dependencies if necessary
try:
    import IPython.display
except ImportError:
    print("IPython is not installed. Installing now...")
    os.system("pip install ipython")
    import IPython.display

# Path to COCO weights file
COCO_WEIGHTS_PATH = r'C:\Users\61449\PycharmProjects\Portfolio_Assessment\mask_rcnn_coco.h5'

# Directory to save logs and model checkpoints
MODEL_DIR = r'C:\Users\61449\PycharmProjects\Portfolio_Assessment\mask_rcnn_logs'

# Configuration for training on the log dataset
class LogConfig(Config):
    NAME = "log"
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1  # Background + log
    STEPS_PER_EPOCH = 100
    DETECTION_MIN_CONFIDENCE = 0.9

# Dataset class
class LogDataset(utils.Dataset):
    def load_logs(self, dataset_dir, subset):
        self.add_class("log", 1, "log")
        # Train or validation dataset
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Use dummy data instead of annotations
        for i in range(1, 11):  # Assuming 10 sample images for training/validation
            image_path = os.path.join(dataset_dir, f'image_{i}.jpg')
            height, width = 128, 128  # Example dimensions for dummy data

            self.add_image(
                "log",
                image_id=i,
                path=image_path,
                width=width,
                height=height,
                annotations=None
            )

    def load_mask(self, image_id):
        # Create dummy masks for the given image
        info = self.image_info[image_id]
        mask = np.zeros([info['height'], info['width'], 1], dtype=np.uint8)
        rr, cc = skimage.draw.rectangle(start=(30, 30), end=(90, 90))  # Dummy rectangle mask
        mask[rr, cc, 0] = 1
        class_ids = np.array([1], dtype=np.int32)
        return mask, class_ids

# Load the training dataset
dataset_train = LogDataset()
dataset_train.load_logs(r'C:\Users\61449\PycharmProjects\Portfolio_Assessment\log_labelled', 'train')
dataset_train.prepare()

# Load the validation dataset
dataset_val = LogDataset()
dataset_val.load_logs(r'C:\Users\61449\PycharmProjects\Portfolio_Assessment\log_labelled', 'val')
dataset_val.prepare()

# Create model in training mode
config = LogConfig()
model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)

# Load weights
model.load_weights(COCO_WEIGHTS_PATH, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

# Train the model
model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=30, layers='heads')
