import os
from torchvision import datasets, transforms

# Define your transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to desired dimensions
    transforms.ToTensor()            # Convert images to tensor
])

# Set the root directory for ImageFolder
root_dir = r'C:\Users\Yusuf Ata\data_science\General\Drone_Project\Fire-Detection'

# Verify the contents of the root directory
print("Contents of Fire-Detection directory:", os.listdir(root_dir))

# Set up your dataset
train_dataset = datasets.ImageFolder(root=root_dir, transform=transform)

# Check the number of classes and images
print(f"Number of classes: {len(train_dataset.classes)}")  # Should print 2 for folders 0 and 1
print(f"Number of images: {len(train_dataset)}")           # Total number of images
print(f"Class names: {train_dataset.classes}")              # Should print ['0', '1']
