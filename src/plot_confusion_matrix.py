import numpy as np
import matplotlib.pyplot as plt

from data_loader import class_names

cm = np.load("models/confusion_matrix.npy")

fig, ax = plt.subplots(figsize=(20, 18))

image = ax.imshow(cm)

fig.colorbar(image, ax=ax)

ax.set(
    xticks=np.arange(len(class_names)),
    yticks=np.arange(len(class_names)),
    xticklabels=class_names,
    yticklabels=class_names,
    xlabel="Predicted Class",
    ylabel="True Class",
    title="Plant Disease Classification Confusion Matrix"
)

plt.setp(
    ax.get_xticklabels(),
    rotation=90,
    ha="center"
)

plt.setp(
    ax.get_yticklabels(),
    rotation=0
)

plt.tight_layout()

plt.savefig(
    "models/confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Confusion matrix image saved to:")
print("models/confusion_matrix.png")