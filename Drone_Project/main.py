import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image  # Import PIL

# Define the CNN model
class FireNet(nn.Module):
    def __init__(self):
        super(FireNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 2)  # Change to 2 for two classes
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 16 * 16)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the trained model
model = FireNet()
model.load_state_dict(torch.load('fire_detection_model.pth', weights_only=True))  # Set weights_only=True
model.eval()

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

# Use OpenCV to access the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    # Preprocess the frame
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)  # Convert the NumPy array to a PIL image
    img = transform(img)  # Apply the transformation
    img = img.unsqueeze(0)  # Add batch dimension

    # Perform inference
    outputs = model(img)
    _, predicted = torch.max(outputs, 1)

    # Display results
    label = 'Fire Detected' if predicted.item() == 1 else 'No Fire'
    color = (0, 0, 255) if label == 'Fire Detected' else (0, 255, 0)
    
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    cv2.imshow('Fire Detection', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
