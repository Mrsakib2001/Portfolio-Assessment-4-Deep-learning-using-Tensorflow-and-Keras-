import labelme2coco

# Define paths
labelme_folder = r'C:\Users\61449\PycharmProjects\Portfolio_Assessment\log_labelled'
save_json_path = r'C:\Users\61449\PycharmProjects\Portfolio_Assessment\log_labelled\annotations_coco.json'

# Convert LabelMe annotations to COCO format
labelme2coco.convert(labelme_folder, save_json_path)
print("Annotations successfully converted to COCO format.")
