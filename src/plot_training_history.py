import matplotlib.pyplot as plt

epochs = list(range(1, 11))

train_accuracy = [
    0.8420,
    0.9046,
    0.9143,
    0.9184,
    0.9189,
    0.9220,
    0.9260,
    0.9369,
    0.9363,
    0.9376
]

val_accuracy = [
    0.9213,
    0.9358,
    0.9287,
    0.9390,
    0.9438,
    0.9385,
    0.9427,
    0.9466,
    0.9560,
    0.9539
]

train_loss = [
    0.5497,
    0.2948,
    0.2655,
    0.2514,
    0.2498,
    0.2443,
    0.2323,
    0.1954,
    0.1944,
    0.1907
]

val_loss = [
    0.2582,
    0.2026,
    0.2243,
    0.1875,
    0.1742,
    0.1878,
    0.1790,
    0.1642,
    0.1478,
    0.1505
]

plt.figure(figsize=(10, 6))

plt.plot(epochs, train_accuracy, marker="o", label="Training Accuracy")
plt.plot(epochs, val_accuracy, marker="o", label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "models/accuracy_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

plt.figure(figsize=(10, 6))

plt.plot(epochs, train_loss, marker="o", label="Training Loss")
plt.plot(epochs, val_loss, marker="o", label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "models/loss_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Training graphs saved successfully.")
print("models/accuracy_curve.png")
print("models/loss_curve.png")