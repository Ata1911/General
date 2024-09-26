import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import transforms

# Load the trained model
model = FireNet()
model.load_state_dict(torch.load('fire_detection_model.pth'))
model.eval()

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Use OpenCV to access the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    # Preprocess the frame
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (64, 64))  # Resize to match model input size
    img = transform(img)
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
