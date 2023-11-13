from keras.models import load_model
from PIL import Image
import numpy as np

# Function to preprocess image
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((150, 150))
    img = np.array(img)
    img = img / 255.0  # Rescale pixel values
    img = np.expand_dims(img, axis=0)
    return img

# Load the trained model
model_path = r"C:\projects\machine learning\single-double\results\model_acc0.9895_2023-08-24-11-28\trained_model.h5"
loaded_model = load_model(model_path)

# Path to the image you want to predict
image_path = r"C:\projects\machine learning\single-double\val\single\page_1335.png"
# image_path = r"C:\projects\machine learning\single-double\val\double\page_975.png"




# Preprocess the image
preprocessed_image = preprocess_image(image_path)

# Make predictions
prediction = loaded_model.predict(preprocessed_image)
print(prediction)
predicted_class = "single page" if prediction > 0.5 else "double page"

print(f"The predicted class is: {predicted_class}")

