import os
import shutil
import json
import random

# Path to the JSON file and image folder
json_file_path = r"C:\Users\86189\Desktop\single-double\project-2-at-2023-08-23-07-05-c2b74c30.json"
image_folder = "C:/Users/86189/Desktop/single-double/"

# Paths for train and validation folders
train_folder = "C:/Users/86189/Desktop/single-double/"
valid_folder = "C:/Users/86189/Desktop/single-double/"

# Create subfolders within train and validation folders
single_train_folder = os.path.join(train_folder, "single_train")
single_val_folder = os.path.join(valid_folder, "single_val")
double_train_folder = os.path.join(train_folder, "double_train")
double_val_folder = os.path.join(valid_folder, "double_val")

# Create the subfolders if they don't exist
os.makedirs(single_train_folder, exist_ok=True)
os.makedirs(single_val_folder, exist_ok=True)
os.makedirs(double_train_folder, exist_ok=True)
os.makedirs(double_val_folder, exist_ok=True)

# Load JSON data
with open(json_file_path, "r") as json_file:
    data = json.load(json_file)

# Split ratio for training and validation
train_ratio = 0.8

# Shuffle the data to ensure randomness
random.shuffle(data)

# Iterate through each entry in the shuffled JSON data
random.shuffle(data)

# Iterate through each entry in the shuffled JSON data
for entry in data:
    print(entry)
    image_filename = entry["image"].split("/")[-1].split("-")[-1]
    source_image_path = os.path.join(image_folder, image_filename)
    choice = entry["choice"]

    if os.path.exists(source_image_path):
        if choice == "single":
            destination_folder = single_train_folder if random.random() < train_ratio else single_val_folder
        else:
            destination_folder = double_train_folder if random.random() < train_ratio else double_val_folder

        destination_image_path = os.path.join(destination_folder, image_filename)

        # Copy image to the respective folder
        shutil.copy(source_image_path, destination_image_path)
    else:
        print(f"Image {image_filename} not found.")

print("Data organization complete.")