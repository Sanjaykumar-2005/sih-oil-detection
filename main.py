from flask import Flask, render_template, request, redirect, flash, url_for
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision.models.segmentation import fcn_resnet50
from torchvision import transforms
from werkzeug.utils import secure_filename

# Initialize Flask app
oil_detection_app = Flask(__name__)

# Load the oil detection model architecture and weights
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the model architecture
model = fcn_resnet50(pretrained=False, aux_loss=False)  # Disable aux_loss
model.classifier[4] = nn.Conv2d(512, 5, kernel_size=1)  # Adjust output layers for 5 classes
model = model.to(device)

# Load the saved weights into the model
state_dict = torch.load('model.pth', map_location=device)

# Remove keys related to aux_classifier from the state_dict
state_dict = {k: v for k, v in state_dict.items() if 'aux_classifier' not in k}

# Load the filtered state_dict
model.load_state_dict(state_dict)
model.eval()

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define color map (as per your model)
reverse_color_map = {
    0: (0, 255, 255),  # Cyan -> Oil
    1: (139, 69, 19),  # Brown -> Ship
    2: (0, 128, 0),    # Green -> Land
    3: (0, 0, 0),      # Black -> Sea
    4: (255, 0, 0)     # Red -> Oil Look-alike
}

# Function to predict the mask
def predict(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to GPU

    with torch.no_grad():
        output = model(image)['out']
        output = torch.argmax(output, dim=1).squeeze().cpu().numpy()  # Get class predictions

    return output

# Function to visualize the predicted mask
def visualize_predictions(prediction):
    h, w = prediction.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_idx, color in reverse_color_map.items():
        rgb_mask[prediction == class_idx] = color
    img = Image.fromarray(rgb_mask)

    # Resize the image to a maximum width of 800px while maintaining aspect ratio
    max_width = 800
    if img.width > max_width:
        ratio = max_width / float(img.width)
        new_height = int((float(img.height) * float(ratio)))
        img = img.resize((max_width, new_height), Image.ANTIALIAS)

    return img
# Function to calculate the area of oil spills
def calculate_oil_spill_area(prediction, pixel_size=10):
    # Count the number of cyan pixels (class index 0)
    num_cyan_pixels = np.sum(prediction == 0)
    # Calculate the area in square meters
    area = num_cyan_pixels * pixel_size
    return area

# Routes
@oil_detection_app.route("/")
@oil_detection_app.route("/home")
def home():
    return render_template("index.html")
 
@oil_detection_app.route('/detect_oil_form', methods=['GET'])
def detect_oil_form():
    return render_template('showcases.html', predicted=False)

@oil_detection_app.route('/detect_oil', methods=['POST'])
def detect_oil():
    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No file part')
            return redirect(request.referrer)

        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join('static/uploads', filename)
            file.save(file_path)

            # Run the model on the uploaded image
            predictions = predict(file_path)

            # Convert predictions to RGB mask
            output_image = visualize_predictions(predictions)
            output_image_path = os.path.join('static/predictions', 'output_' + filename)
            output_image.save(output_image_path)
            oil_spill_area = calculate_oil_spill_area(predictions, pixel_size=10)

            return render_template('blog-single.html', original_image=file_path, output_image=output_image_path, summary=oil_spill_area)

if __name__ == "__main__":
    oil_detection_app.secret_key = 'super secret key'
    oil_detection_app.run(debug=True)
