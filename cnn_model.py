import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import csv

# Define the base directory paths
base_dir = "C:\\Users\\61449\\PycharmProjects\\Portfolio_Assessment"
train_dir = os.path.join(base_dir, 'corrosion')
test_dir = 'C:\\Users\\61449\\PycharmProjects\\Portfolio_Assessment\\test_images'


# Image specifications
img_height, img_width = 150, 150
batch_size = 32

# Data generators for normalization
train_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Load train and test sets
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary'
)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=1,
    class_mode='binary',
    shuffle=False
)

# CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(train_generator, epochs=10, validation_data=test_generator)

# Evaluate model
test_loss, test_accuracy = model.evaluate(test_generator, steps=len(test_generator))
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

# Save model
model_path = os.path.join(base_dir, 'cnn_model.h5')
model.save(model_path)
print(f"Model saved to {model_path}")

# Generate and save predictions
print("Generating predictions...")
csv_path = os.path.join(base_dir, 'cnn_test', 'cnn_predictions.csv')
os.makedirs(os.path.dirname(csv_path), exist_ok=True)  # Ensure the directory exists

with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Image', 'Predicted Class', 'Actual Class'])

    for i in range(len(test_generator)):
        img, label = test_generator[i]
        prediction = model.predict(img)[0][0]
        predicted_class = 'rust' if prediction > 0.5 else 'no rust'
        actual_class = 'rust' if label[0] == 1 else 'no rust'
        writer.writerow([f'Image {i + 1}', predicted_class, actual_class])

print(f"Predictions saved to {csv_path}.")
