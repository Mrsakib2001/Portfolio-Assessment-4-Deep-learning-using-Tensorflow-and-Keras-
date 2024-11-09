import os
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import csv

# Define paths
base_dir = "C:\\Users\\61449\\PycharmProjects\\Portfolio_Assessment"
train_dir = os.path.join(base_dir, 'corrosion')
test_dir = 'C:\\Users\\61449\\PycharmProjects\\Portfolio_Assessment\\test_images'

resnet50_test_dir = os.path.join(base_dir, 'resnet50_test')
os.makedirs(resnet50_test_dir, exist_ok=True)

# Image specs
img_height, img_width = 150, 150
batch_size = 32

# Data generators
train_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

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

# Load ResNet50
resnet_base = ResNet50(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# Freeze base layers
for layer in resnet_base.layers:
    layer.trainable = False

# Add custom layers
x = Flatten()(resnet_base.output)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)
model = Model(inputs=resnet_base.input, outputs=output)

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(train_generator, epochs=10, validation_data=test_generator)

# Evaluate model
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"ResNet50 Model Test Accuracy: {test_accuracy * 100:.2f}%")

# Save model
model_path = os.path.join(base_dir, 'resnet50_model.h5')
model.save(model_path)
print(f"Model saved to {model_path}")

# Generate and save predictions
print("Generating predictions...")
csv_path = os.path.join(resnet50_test_dir, 'resnet50_predictions.csv')

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
