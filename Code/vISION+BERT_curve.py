import matplotlib.pyplot as plt

# Data from the epochs
epochs = list(range(1, 11))

train_loss = [0.4816, 0.3298, 0.2439, 0.2095, 0.1692, 0.1585, 0.1313, 0.1215, 0.1055, 0.0980]
val_loss = [0.3509, 0.2607, 0.2896, 0.2921, 0.2415, 0.2707, 0.2452, 0.1997, 0.3067, 0.2302]

train_acc = [0.8443, 0.8855, 0.9143, 0.9259, 0.9401, 0.9449, 0.9547, 0.9567, 0.9615, 0.9658]
val_acc = [0.8621, 0.9074, 0.8929, 0.9093, 0.9310, 0.9220, 0.9256, 0.9437, 0.9129, 0.9383]

# Plot Loss and Accuracy
plt.figure(figsize=(12, 5))

# Loss Curve
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, label='Training Loss', marker='o')
plt.plot(epochs, val_loss, label='Validation Loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.grid(True)

# Accuracy Curve
plt.subplot(1, 2, 2)
plt.plot(epochs, train_acc, label='Training Accuracy', marker='o')
plt.plot(epochs, val_acc, label='Validation Accuracy', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve')
plt.legend()
plt.grid(True)

plt.tight_layout()

# Save the figure
plt.savefig("ABLERCURVE.png", dpi=300)

# Show the plot
plt.show()
