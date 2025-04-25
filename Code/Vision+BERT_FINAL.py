import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from transformers import BertTokenizer, BertModel
import timm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import os

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
data = pd.read_csv('/home/anshul/DIP_Project/DATA.csv')

# Text creation
def create_text_description(row):
    return f"A {row['age']} year old {row['sex']} with a lesion at {row['localization']}."

# Encode labels
label_encoder = LabelEncoder()
data['label_encoded'] = label_encoder.fit_transform(data['dx_cat'])
class_names = [str(label) for label in label_encoder.classes_]

# Image transformations with augmentation
transform_train = transforms.Compose([ 
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

transform_val = transforms.Compose([ 
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# Dataset
class SkinLesionDataset(Dataset):
    def __init__(self, df, tokenizer, transform=None):
        self.df = df
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row['image_path']
        text = create_text_description(row)
        label = row['label_encoded']

        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        text_inputs = self.tokenizer(text, return_tensors="pt", padding='max_length', max_length=64, truncation=True)
        text_inputs = {k: v.squeeze(0) for k, v in text_inputs.items()}

        return image, text_inputs, torch.tensor(label)

# Split data (80% train, 10% validation, 10% test)
train_df, temp_df = train_test_split(data, test_size=0.2, stratify=data['label_encoded'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label_encoded'], random_state=42)

# Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Dataloaders
train_dataset = SkinLesionDataset(train_df, tokenizer, transform=transform_train)
val_dataset = SkinLesionDataset(val_df, tokenizer, transform=transform_val)
test_dataset = SkinLesionDataset(test_df, tokenizer, transform=transform_val)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)
test_loader = DataLoader(test_dataset, batch_size=16)

# Model
class ALBEFClassifier(nn.Module):
    def __init__(self, image_encoder, text_encoder, hidden_dim=768, num_classes=7):
        super(ALBEFClassifier, self).__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, image, text_input):
        image_feat = self.image_encoder(image)
        text_output = self.text_encoder(
            input_ids=text_input['input_ids'],
            attention_mask=text_input['attention_mask']
        )
        text_feat = text_output.last_hidden_state[:, 0, :]
        combined = torch.cat((image_feat, text_feat), dim=1)
        fused = self.fusion(combined)
        out = self.classifier(fused)
        return out

# Load encoders
image_encoder = timm.create_model('vit_base_patch16_224', pretrained=True)
image_encoder.head = nn.Identity()
text_encoder = BertModel.from_pretrained("bert-base-uncased")

# Model instance
model = ALBEFClassifier(image_encoder, text_encoder, hidden_dim=768, num_classes=len(class_names)).to(device)

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

# Early stopping setup
best_val_loss = float('inf')
patience = 5
patience_counter = 0

# Tracking predictions for combined ROC curve
y_true_combined = []
y_pred_combined = []
y_pred_prob_combined = []

# Training loop
def train(model, loader):
    model.train()
    running_loss, correct_predictions, total_samples = 0, 0, 0
    for images, texts, labels in loader:
        images, labels = images.to(device), labels.to(device)
        texts = {k: v.to(device) for k, v in texts.items()}

        optimizer.zero_grad()
        outputs = model(images, texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct_predictions += torch.sum(preds == labels)
        total_samples += labels.size(0)

    avg_loss = running_loss / len(loader)
    accuracy = correct_predictions / total_samples
    train_losses.append(avg_loss)
    train_accuracies.append(accuracy.item())
    return avg_loss, accuracy.item()

# Evaluation
def evaluate(model, loader):
    model.eval()
    y_true, y_pred, y_pred_prob = [], [], []
    running_loss = 0
    with torch.no_grad():
        for images, texts, labels in loader:
            images, labels = images.to(device), labels.to(device)
            texts = {k: v.to(device) for k, v in texts.items()}
            outputs = model(images, texts)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_pred_prob.extend(torch.softmax(outputs, dim=1).cpu().numpy())

    avg_loss = running_loss / len(loader)
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    return avg_loss, acc, precision, recall, f1, y_true, y_pred, np.array(y_pred_prob)

# Run training and evaluation
for epoch in range(10):
    print(f"\nEpoch {epoch + 1}")
    train_loss, train_acc = train(model, train_loader)
    print(f"Training Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

    val_loss, val_acc, precision, recall, f1, y_true, y_pred, y_pred_prob = evaluate(model, val_loader)
    print(f"Validation - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    # Collect data for combined ROC curve
    y_true_combined.extend(y_true)
    y_pred_combined.extend(y_pred)
    y_pred_prob_combined.extend(y_pred_prob)

    # Early stopping logic
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered. Training stopped.")
            break

    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

# After training, get the overall classification report on the test set
test_loss, test_acc, precision, recall, f1, y_true, y_pred, y_pred_prob = evaluate(model, test_loader)
print("\nTest Evaluation:")
print(f"Test - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
print(classification_report(y_true, y_pred, target_names=class_names))

# Save the loss and accuracy curves
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig("BERT_loss_curve.png")

plt.subplot(1, 2, 2)
plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("BERT_accuracy_curve.png")

# Save the ROC curve
fpr, tpr, _ = roc_curve(y_true_combined, np.array(y_pred_prob_combined)[:, 1])
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='b', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.savefig("BERT_roc_curve.png")

print("Plots saved as 'BERT_loss_curve.png', 'BERT_accuracy_curve.png', and 'BERT_roc_curve.png'")
