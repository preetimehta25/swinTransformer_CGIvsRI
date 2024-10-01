import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import SwinForImageClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_curve, auc
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline

# Define the custom dataset
class CustomImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# Paths to images
dataset_dir = Path('dataset')
real_images = list(dataset_dir.glob('real/*'))
computer_generated_images = list(dataset_dir.glob('computer_generated/*'))

# Create labels (0 for real, 1 for computer-generated)
image_paths = real_images + computer_generated_images
labels = [0] * len(real_images) + [1] * len(computer_generated_images)

# Train-validation-test split
train_paths, temp_paths, train_labels, temp_labels = train_test_split(image_paths, labels, test_size=0.2, random_state=42)
val_paths, test_paths, val_labels, test_labels = train_test_split(temp_paths, temp_labels, test_size=0.5, random_state=42)

# Data augmentation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
])

# Create datasets with augmentation
train_dataset = CustomImageDataset(train_paths, train_labels, transform=transform)
val_dataset = CustomImageDataset(val_paths, val_labels, transform=transform)
test_dataset = CustomImageDataset(test_paths, test_labels, transform=transform)

# Data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load pre-trained Swin Transformer model for feature extraction
model_name = 'microsoft/swin-tiny-patch4-window7-224'
model = SwinForImageClassification.from_pretrained(model_name)
model.classifier = torch.nn.Identity()  # Remove the final classifier layer
model = model.to(device)

# Feature extraction
def extract_features(data_loader, model, device):
    model.eval()
    features_list = []
    labels_list = []
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            features = model(images).logits
            features_list.append(features.cpu().numpy())
            labels_list.extend(labels.numpy())
    features = np.vstack(features_list)
    labels = np.array(labels_list)
    return features, labels


# Extract features from train, validation, and test sets
train_features, train_labels = extract_features(train_loader, model, device)
val_features, val_labels = extract_features(val_loader, model, device)
test_features, test_labels = extract_features(test_loader, model, device)

# Standardize the features
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
val_features = scaler.transform(val_features)
test_features = scaler.transform(test_features)

# Train an SVM classifier
svm_classifier = SVC(kernel='rbf', probability=True, random_state=42)
svm_classifier.fit(train_features, train_labels)

# Evaluate the SVM classifier
val_predictions = svm_classifier.predict(val_features)
test_predictions = svm_classifier.predict(test_features)

# Train the pipeline
num_epochs = 20
train_accuracies = []
val_accuracies = []

for epoch in range(num_epochs):
    classifier.fit(train_features, train_labels)
    
    # Evaluate on training data
    train_predictions = classifier.predict(train_features)
    train_accuracy = accuracy_score(train_labels, train_predictions)
    train_accuracies.append(train_accuracy)

    # Evaluate on validation data
    val_predictions = classifier.predict(val_features)
    val_accuracy = accuracy_score(val_labels, val_predictions)
    val_accuracies.append(val_accuracy)

    print(f"Epoch [{epoch+1}/{num_epochs}], Training Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {val_accuracy:.2f}%")

# Final evaluation on test set
test_predictions = classifier.predict(test_features)
test_accuracy = accuracy_score(test_labels, test_predictions)
precision, recall, fscore, _ = precision_recall_fscore_support(test_labels, test_predictions, average='binary')
fpr, tpr, _ = roc_curve(test_labels, test_predictions)
roc_auc = auc(fpr, tpr)

print(f'Final Test Accuracy: {test_accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1-Score: {fscore:.2f}')
print(f'AUC: {roc_auc:.2f}')

# Plotting Loss and Accuracy
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy Over Epochs')
plt.legend()

plt.tight_layout()
plt.savefig('training_accuracy.png', dpi=700)
plt.show()

# Plot ROC Curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.savefig('roc_curve.png', dpi=700)
plt.show()

# t-SNE visualization
features_tsne = TSNE(n_components=2, random_state=42).fit_transform(test_features)
plt.figure(figsize=(10, 8))
scatter = plt.scatter(features_tsne[:, 0], features_tsne[:, 1], c=test_labels, cmap='viridis', alpha=0.6)
plt.colorbar(scatter)
plt.title('t-SNE Plot of Features')
plt.xlabel('t-SNE Feature 1')
plt.ylabel('t-SNE Feature 2')
plt.savefig('tsne_plot.png', dpi=700)
plt.show()
