import os
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.preprocessing import LabelEncoder, label_binarize
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import models, transforms
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

# Load dataset
data = pd.read_csv('/home/anshul/DIP_Project/DATA.csv')

# Encode labels
label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['dx_cat'])  # change 'dx_cat' if needed
class_names = label_encoder.classes_

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# Transforms
transform = transforms.Compose([
    transforms.Resize((img_h, img_w)),
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

# Calculate class weights based on the frequency of each class in the training data
class_weights = 1. / train_data['label'].value_counts().values
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

# Modify the criterion to use class weights
criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)

# Model initializer
def initialise_model(model_name, num_classes, feature_extract=False, use_pretrained=True):
    if model_name == 'densenet_pret':
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1 if use_pretrained else None)
        if feature_extract:
            for param in model.parameters():
                param.requires_grad = False
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
    else:
        raise ValueError(f"Model {model_name} not recognized")
    return model

# Training function
def train_and_validate_model(model, train_loader, val_loader, test_loader, criterion, patience, optimizer, device, epochs, model_filename, verbose=False):
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

    # Plot and save loss and accuracy curves
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(epochs), train_loss_history, label='Train Loss')
    plt.plot(range(epochs), val_loss_history, label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(epochs), train_acc_history, label='Train Accuracy')
    plt.plot(range(epochs), val_acc_history, label='Validation Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('loss_accuracy_curves.png')
    print("Loss and accuracy curves saved to loss_accuracy_curves.png")

    return model, train_loss_history, val_loss_history, train_acc_history, val_acc_history, all_labels, all_preds, all_probs

# Evaluation
def evaluate_and_save_metrics(model, test_loader, device, class_names):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('11_densenet_confusion_matrix.png')
    print("Confusion matrix saved to 11_densenet_confusion_matrix.png")
    plt.close()

    # ROC AUC
    all_labels_bin = label_binarize(all_labels, classes=range(len(class_names)))
    roc_auc = roc_auc_score(all_labels_bin, all_probs, average='weighted', multi_class='ovr')
    print(f"ROC AUC Score: {roc_auc:.4f}")

    # ROC Curve
    fpr, tpr, _ = roc_curve(all_labels_bin.ravel(), np.array(all_probs).ravel())
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='b', label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('11_densenet_curve_roc.png')
    print("ROC curve saved to final_densenet_curve_roc.png")
    plt.close()

    # Save predictions
    predictions_df = pd.DataFrame({
        'True Labels': all_labels,
        'Predictions': all_preds,
        'Max Probabilities': np.max(all_probs, axis=1)
    })
    predictions_df.to_csv('11_densenet_predictions.csv', index=False)
    print("Predictions saved to 11_densenet_predictions.csv")

# Set parameters
model_temp = 'densenet_pret'
num_classes = len(class_names)
feature_extract = False
use_pretrained = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 20
patience = 3
model_filename = 'final_densenet_pret.pth'

# Initialize and train model
model = initialise_model(model_temp, num_classes, feature_extract, use_pretrained)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

model, train_loss, val_loss, train_acc, val_acc, all_labels, all_preds, all_probs = train_and_validate_model(
    model, train_loader, val_loader, test_loader, criterion, patience, optimizer, device, epochs, model_filename, verbose=True)

# Final evaluation
evaluate_and_save_metrics(model, test_loader, device, class_names)
