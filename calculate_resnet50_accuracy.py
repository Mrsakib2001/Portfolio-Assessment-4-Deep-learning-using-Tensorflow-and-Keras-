import csv

# Path to the CSV file for ResNet50 predictions
csv_path = "C:\\Users\\61449\\PycharmProjects\\Portfolio_Assessment\\resnet50_test\\resnet50_predictions.csv"

# Initialize counters
correct_predictions = 0
total_predictions = 0

# Open and read the CSV file
with open(csv_path, 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header row

    # Iterate through each row in the CSV
    for row in reader:
        predicted_class = row[1]  # Predicted class ("rust" or "no rust")
        actual_class = row[2]  # Actual class ("rust" or "no rust")

        # Compare predicted and actual class
        if predicted_class == actual_class:
            correct_predictions += 1
        total_predictions += 1

# Calculate accuracy
accuracy = (correct_predictions / total_predictions) * 100
print(f"ResNet50 Model Test Accuracy: {accuracy:.2f}%")
