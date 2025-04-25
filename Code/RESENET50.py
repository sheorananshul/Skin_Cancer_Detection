import os
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from tqdm import tqdm

# Load dataset
data = pd.read_csv('/home/anshul/DIP_Project/DATA.csv')

# Encode labels
label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['dx_cat'])  # change 'dx_cat' if needed
class_names = label_encoder.classes_

# Image parameters
img_h, img_w = 224, 224
norm_means = [0.77148203, 0.55764165, 0.58345652]
norm_std = [0.12655577, 0.14245141, 0.15189891]

# Dataset class
class HAM10000Dataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_path = self.dataframe.loc[idx, 'image_path']
        label = self.dataframe.loc[idx, 'label']
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# Data augmentation and normalizer
transform = transforms.Compose([
    transforms.Resize((img_h, img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomVerticalFlip(),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),  # Added affine transformation
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Added color jitter
    transforms.ToTensor(),
    transforms.Normalize(mean=norm_means, std=norm_std)
])

# 80-10-10 Split
train_data, temp_data = train_test_split(data, test_size=0.2, stratify=data['label'], random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, stratify=temp_data['label'], random_state=42)

# Data balancing (oversample minority classes in training data)
max_count = train_data['label'].value_counts().max()
balanced_train_data = pd.concat([ 
    train_data[train_data['label'] == label].sample(max_count, replace=True, random_state=42) 
    for label in train_data['label'].unique()
], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

# Create datasets
train_dataset = HAM10000Dataset(balanced_train_data, transform=transform)
val_dataset = HAM10000Dataset(val_data, transform=transform)
test_dataset = HAM10000Dataset(test_data, transform=transform)

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Model initializer for ResNet50 with Dropout layer
def initialise_model(model_name, num_classes, feature_extract=True, use_pretrained=True):
    if model_name == 'resnet50':
        model = models.resnet50(pretrained=use_pretrained)
        
        # Freeze parameters if feature extraction
        if feature_extract:
            for param in model.parameters():
                param.requires_grad = False

        # Replace the final fully connected layer
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, num_classes),
            nn.Dropout(0.5)  # Adding Dropout for regularization
        )
    
    else:
        raise ValueError(f"Model {model_name} not supported.")
    
    return model

# Training function with L2 Regularization (weight decay)
def train_and_validate_model(model, train_loader, val_loader, criterion, patience, optimizer, device, epochs, model_filename, verbose=False):
    best_val_loss = float('inf')
    early_stopping_counter = 0
    train_loss_history, val_loss_history = [], []
    train_acc_history, val_acc_history = [], []
    all_labels, all_preds, all_probs = [], [], []

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_train_loss = total_loss / len(train_loader)
        train_accuracy = correct / total
        train_loss_history.append(avg_train_loss)
        train_acc_history.append(train_accuracy)

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_total
        val_loss_history.append(avg_val_loss)
        val_acc_history.append(val_accuracy)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_filename)
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print("Early stopping triggered.")
                break

        if verbose:
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

    return model, train_loss_history, val_loss_history, train_acc_history, val_acc_history, all_labels, all_preds, all_probs

# Plotting training and validation curves
def plot_training_validation_curve(train_loss_history, val_loss_history, train_acc_history, val_acc_history):
    # Plot Loss curve
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(train_loss_history) + 1), train_loss_history, label='Training Loss')
    plt.plot(range(1, len(val_loss_history) + 1), val_loss_history, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('50training_validation_loss_curve.png')
    plt.show()

    # Plot Accuracy curve
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(train_acc_history) + 1), train_acc_history, label='Training Accuracy')
    plt.plot(range(1, len(val_acc_history) + 1), val_acc_history, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig('50training_validation_accuracy_curve.png')
    plt.show()

# Evaluation and metric saving function
def evaluate_and_save_metrics(model, test_loader, device, class_names):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(probs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(all_labels, all_preds)

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1 Score: {f1:.4f}")

    # Confusion Matrix Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig('50confusion_matrix.png')
    plt.show()

    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
        'Value': [accuracy, precision, recall, f1]
    })
    metrics_df.to_csv('50test_metrics.csv', index=False)

# Model setup and training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = initialise_model('resnet50', num_classes=len(class_names), feature_extract=True, use_pretrained=True).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)  # Adding L2 regularization here
epochs = 25
model_filename = 'best_resnet50_model.pth'
patience = 3

# Train and validate model
start_time = time.time()
model, train_loss_history, val_loss_history, train_acc_history, val_acc_history, all_labels, all_preds, all_probs = train_and_validate_model(
    model, train_loader, val_loader, criterion, patience, optimizer, device, epochs, model_filename, verbose=True)
end_time = time.time()

# Print elapsed time
print(f"Training completed in {(end_time - start_time) / 60:.2f} minutes.")

# Plot curves
plot_training_validation_curve(train_loss_history, val_loss_history, train_acc_history, val_acc_history)

# Load best model for evaluation
model.load_state_dict(torch.load(model_filename))

# Evaluate and save metrics
evaluate_and_save_metrics(model, test_loader, device, class_names)
