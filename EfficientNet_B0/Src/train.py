from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

TRAIN_DIR = PROJECT_DIR / "Dataset_Split" / "train"
VAL_DIR = PROJECT_DIR / "Dataset_Split" / "validation"
TEST_DIR = PROJECT_DIR / "Dataset_Split" / "test"

MODEL_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = MODEL_DIR / "efficientnet_b0_good_bad.pth"


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = 224
BATCH_SIZE = 16
NUM_EPOCHS = 15
LEARNING_RATE = 0.001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("EfficientNet-B0 GOOD/BAD Classification")
print("=" * 60)

print(f"Project directory : {PROJECT_DIR}")
print(f"Device            : {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU               : {torch.cuda.get_device_name(0)}")


# ============================================================
# DATA TRANSFORMS
# ============================================================

# Training augmentation.
#
# IMPORTANT:
# These are only CNN training augmentations.
# They are NOT your barcode preprocessing pipeline.
#
# We are NOT doing:
# CLAHE
# Otsu
# sharpening
# denoising
# upscaling
# perspective correction

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomRotation(
        degrees=10
    ),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# Validation and test:
# NO random augmentation.

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# ============================================================
# LOAD DATASETS
# ============================================================

print("\nLoading datasets...")

train_dataset = datasets.ImageFolder(
    root=TRAIN_DIR,
    transform=train_transform,
)

val_dataset = datasets.ImageFolder(
    root=VAL_DIR,
    transform=eval_transform,
)

test_dataset = datasets.ImageFolder(
    root=TEST_DIR,
    transform=eval_transform,
)


print("\nClass mapping:")
print(train_dataset.class_to_idx)

print("\nDataset sizes:")
print(f"Train      : {len(train_dataset)}")
print(f"Validation : {len(val_dataset)}")
print(f"Test       : {len(test_dataset)}")


# ============================================================
# DATA LOADERS
# ============================================================

# num_workers=0 is intentionally used here because you are
# working on Windows and it avoids multiprocessing issues.

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


# ============================================================
# LOAD PRETRAINED EFFICIENTNET-B0
# ============================================================

print("\nLoading pretrained EfficientNet-B0...")

weights = EfficientNet_B0_Weights.DEFAULT

model = efficientnet_b0(
    weights=weights
)


# ============================================================
# FREEZE BACKBONE
# ============================================================

print("Freezing EfficientNet-B0 backbone...")

for parameter in model.features.parameters():
    parameter.requires_grad = False


# ============================================================
# REPLACE CLASSIFICATION HEAD
# ============================================================

# EfficientNet-B0 pretrained model:
#
# 1280 features
#       ↓
# ImageNet classifier
#       ↓
# 1000 classes
#
# We replace the final layer with:
#
# 1280 features
#       ↓
# GOOD / BAD
#       ↓
# 2 classes

num_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    num_features,
    2,
)

model = model.to(DEVICE)


print("\nFinal classifier:")
print(model.classifier)


# ============================================================
# LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# OPTIMIZER
# ============================================================

# Only train parameters that require gradients.
# Since backbone is frozen, this means mainly the classifier.

optimizer = torch.optim.AdamW(
    filter(
        lambda parameter: parameter.requires_grad,
        model.parameters()
    ),
    lr=LEARNING_RATE,
    weight_decay=1e-4,
)


# ============================================================
# TRAIN FUNCTION
# ============================================================

def train_one_epoch():
    model.train()

    total_loss = 0.0
    all_labels = []
    all_predictions = []

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item() *
            images.size(0)
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_labels.extend(
            labels.detach().cpu().numpy()
        )

        all_predictions.extend(
            predictions.detach().cpu().numpy()
        )

    average_loss = (
        total_loss /
        len(train_dataset)
    )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    return average_loss, accuracy


# ============================================================
# VALIDATION FUNCTION
# ============================================================

def validate():
    model.eval()

    total_loss = 0.0
    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            total_loss += (
                loss.item() *
                images.size(0)
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

    average_loss = (
        total_loss /
        len(val_dataset)
    )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    return (
        average_loss,
        accuracy,
        precision,
        recall,
        f1,
    )


# ============================================================
# TRAINING LOOP
# ============================================================

best_val_f1 = -1.0

print("\n")
print("=" * 60)
print("STARTING TRAINING")
print("=" * 60)


for epoch in range(NUM_EPOCHS):

    train_loss, train_accuracy = train_one_epoch()

    (
        val_loss,
        val_accuracy,
        val_precision,
        val_recall,
        val_f1,
    ) = validate()

    print("\n" + "-" * 60)

    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS}"
    )

    print(
        f"Train Loss      : {train_loss:.4f}"
    )

    print(
        f"Train Accuracy  : {train_accuracy:.4f}"
    )

    print(
        f"Validation Loss : {val_loss:.4f}"
    )

    print(
        f"Validation Acc  : {val_accuracy:.4f}"
    )

    print(
        f"Validation Prec : {val_precision:.4f}"
    )

    print(
        f"Validation Recall: {val_recall:.4f}"
    )

    print(
        f"Validation F1   : {val_f1:.4f}"
    )

    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_f1 > best_val_f1:

        best_val_f1 = val_f1

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "class_to_idx": train_dataset.class_to_idx,
                "validation_f1": best_val_f1,
            },
            BEST_MODEL_PATH,
        )

        print(
            f"*** Best model saved "
            f"(Validation F1 = {best_val_f1:.4f})"
        )


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\n")
print("=" * 60)
print("LOADING BEST MODEL")
print("=" * 60)

checkpoint = torch.load(
    BEST_MODEL_PATH,
    map_location=DEVICE,
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

print(
    f"Best epoch       : {checkpoint['epoch']}"
)

print(
    f"Best validation F1: "
    f"{checkpoint['validation_f1']:.4f}"
)


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

print("\n")
print("=" * 60)
print("FINAL TEST EVALUATION")
print("=" * 60)

model.eval()

all_test_labels = []
all_test_predictions = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_test_labels.extend(
            labels.numpy()
        )

        all_test_predictions.extend(
            predictions.cpu().numpy()
        )


# ============================================================
# TEST METRICS
# ============================================================

test_accuracy = accuracy_score(
    all_test_labels,
    all_test_predictions
)

test_precision = precision_score(
    all_test_labels,
    all_test_predictions,
    zero_division=0
)

test_recall = recall_score(
    all_test_labels,
    all_test_predictions,
    zero_division=0
)

test_f1 = f1_score(
    all_test_labels,
    all_test_predictions,
    zero_division=0
)


print("\nTest Results:")

print(
    f"Accuracy  : {test_accuracy:.4f}"
)

print(
    f"Precision : {test_precision:.4f}"
)

print(
    f"Recall    : {test_recall:.4f}"
)

print(
    f"F1 Score  : {test_f1:.4f}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

class_names = [
    name
    for name, index in sorted(
        train_dataset.class_to_idx.items(),
        key=lambda item: item[1]
    )
]

report = classification_report(
    all_test_labels,
    all_test_predictions,
    target_names=class_names,
    zero_division=0,
)

print("\nClassification Report:")
print(report)


# Save report
report_path = RESULTS_DIR / "classification_report.txt"

with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "EfficientNet-B0 GOOD/BAD Classification\n\n"
    )

    file.write(
        f"Test Accuracy  : {test_accuracy:.4f}\n"
    )

    file.write(
        f"Test Precision : {test_precision:.4f}\n"
    )

    file.write(
        f"Test Recall    : {test_recall:.4f}\n"
    )

    file.write(
        f"Test F1        : {test_f1:.4f}\n\n"
    )

    file.write(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_test_labels,
    all_test_predictions,
)

print("\nConfusion Matrix:")
print(cm)


plt.figure(
    figsize=(6, 5)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=class_names,
    yticklabels=class_names,
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("EfficientNet-B0 GOOD/BAD")

plt.tight_layout()

cm_path = RESULTS_DIR / "confusion_matrix.png"

plt.savefig(
    cm_path,
    dpi=200,
)

plt.close()


print("\n")
print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(f"Model saved to:")
print(BEST_MODEL_PATH)

print(f"\nResults saved to:")
print(RESULTS_DIR)