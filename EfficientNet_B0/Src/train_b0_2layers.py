from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(
    __file__
).resolve().parent.parent

WORKSPACE_DIR = PROJECT_DIR.parent

DATASET_DIR = (
    WORKSPACE_DIR
    / "Dataset_for_Training"
    / "Dataset_Split"
)

TRAIN_DIR = (
    DATASET_DIR
    / "train"
)

VAL_DIR = (
    DATASET_DIR
    / "validation"
)

TEST_DIR = (
    DATASET_DIR
    / "test"
)

MODEL_DIR = (
    PROJECT_DIR
    / "models"
)

RESULTS_DIR = (
    PROJECT_DIR
    / "results"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# OUTPUT FILE
# ============================================================

BEST_MODEL_PATH = (
    MODEL_DIR
    / "efficientnet_b0_good_bad_2layers.pth"
)


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = 224
BATCH_SIZE = 16

NUM_EPOCHS = 15

# Keep the same learning rate as your existing model
LEARNING_RATE = 0.001

WEIGHT_DECAY = 1e-4


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print(
    "EfficientNet-B0 GOOD/BAD Classification"
)
print(
    "Experiment: B0 + 2 Extra Layers"
)
print("=" * 70)

print(
    f"Project directory : {PROJECT_DIR}"
)

print(
    f"Device            : {DEVICE}"
)

if torch.cuda.is_available():

    print(
        "GPU               : "
        f"{torch.cuda.get_device_name(0)}"
    )


# ============================================================
# DATA TRANSFORMS
# ============================================================

# Training augmentation
#
# These are only CNN training augmentations.
#
# They are NOT barcode preprocessing.
#
# No:
# - CLAHE
# - Otsu
# - sharpening
# - denoising
# - upscaling
# - perspective correction

train_transform = transforms.Compose(
    [

        transforms.Resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE
            )
        ),

        transforms.RandomRotation(
            degrees=10
        ),

        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406
            ],
            std=[
                0.229,
                0.224,
                0.225
            ]
        ),
    ]
)


# ============================================================
# VALIDATION / TEST TRANSFORM
# ============================================================

eval_transform = transforms.Compose(
    [

        transforms.Resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE
            )
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406
            ],
            std=[
                0.229,
                0.224,
                0.225
            ]
        ),
    ]
)


# ============================================================
# DATASETS
# ============================================================

print(
    "\nLoading datasets..."
)

train_dataset = datasets.ImageFolder(
    root=TRAIN_DIR,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    root=VAL_DIR,
    transform=eval_transform
)

test_dataset = datasets.ImageFolder(
    root=TEST_DIR,
    transform=eval_transform
)


print(
    "\nClass mapping:"
)

print(
    train_dataset.class_to_idx
)


print(
    "\nDataset sizes:"
)

print(
    f"Train      : {len(train_dataset)}"
)

print(
    f"Validation : {len(val_dataset)}"
)

print(
    f"Test       : {len(test_dataset)}"
)


# ============================================================
# DATA LOADERS
# ============================================================

# num_workers=0 is retained from your existing code
# because you are running on Windows.

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# LOAD PRETRAINED EFFICIENTNET-B0
# ============================================================

print(
    "\nLoading pretrained EfficientNet-B0..."
)

weights = EfficientNet_B0_Weights.DEFAULT

model = efficientnet_b0(
    weights=weights
)


# ============================================================
# FREEZE B0 BACKBONE
# ============================================================

print(
    "Freezing EfficientNet-B0 backbone..."
)

for parameter in model.features.parameters():

    parameter.requires_grad = False


# ============================================================
# REPLACE CLASSIFIER
# ============================================================
#
# Original:
#
# 1280
#   ↓
# 1000 ImageNet classes
#
# New:
#
# 1280
#   ↓
# Linear 256
#   ↓
# ReLU
#   ↓
# Linear 64
#   ↓
# ReLU
#   ↓
# Linear 2
#   ↓
# BAD / GOOD
#
# Only these new layers are trainable.
# ============================================================

num_features = (
    model.classifier[1].in_features
)


model.classifier = nn.Sequential(

    # ========================================================
    # EXTRA LAYER 1
    # ========================================================

    nn.Linear(
        num_features,
        256
    ),

    nn.ReLU(
        inplace=True
    ),


    # ========================================================
    # EXTRA LAYER 2
    # ========================================================

    nn.Linear(
        256,
        64
    ),

    nn.ReLU(
        inplace=True
    ),


    # ========================================================
    # FINAL CLASSIFICATION LAYER
    # ========================================================

    nn.Linear(
        64,
        2
    )
)


# ============================================================
# MOVE MODEL TO DEVICE
# ============================================================

model = model.to(
    DEVICE
)


# ============================================================
# PRINT CLASSIFIER
# ============================================================

print(
    "\nNew classifier:"
)

print(
    model.classifier
)


# ============================================================
# CHECK TRAINABLE PARAMETERS
# ============================================================

print(
    "\nTrainable parameters:"
)

trainable_parameters = 0

for name, parameter in model.named_parameters():

    if parameter.requires_grad:

        parameter_count = (
            parameter.numel()
        )

        trainable_parameters += (
            parameter_count
        )

        print(
            f"{name}: "
            f"{parameter_count:,}"
        )


print(
    "\nTotal trainable parameters:"
)

print(
    f"{trainable_parameters:,}"
)


# ============================================================
# LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# OPTIMIZER
# ============================================================
#
# Gradient-based learning:
#
# loss
#   ↓
# backward()
#   ↓
# gradients
#   ↓
# AdamW
#   ↓
# weight update
#
# ============================================================

optimizer = torch.optim.AdamW(

    filter(
        lambda parameter:
            parameter.requires_grad,
        model.parameters()
    ),

    lr=LEARNING_RATE,

    weight_decay=WEIGHT_DECAY
)


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_one_epoch():

    model.train()

    total_loss = 0.0

    all_labels = []
    all_predictions = []


    for images, labels in train_loader:

        images = images.to(
            DEVICE
        )

        labels = labels.to(
            DEVICE
        )


        # ----------------------------------------------------
        # Clear old gradients
        # ----------------------------------------------------

        optimizer.zero_grad()


        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        outputs = model(
            images
        )


        # ----------------------------------------------------
        # Calculate loss
        # ----------------------------------------------------

        loss = criterion(
            outputs,
            labels
        )


        # ----------------------------------------------------
        # BACKPROPAGATION
        # ----------------------------------------------------

        loss.backward()


        # ----------------------------------------------------
        # GRADIENT-BASED WEIGHT UPDATE
        # ----------------------------------------------------

        optimizer.step()


        total_loss += (
            loss.item()
            * images.size(0)
        )


        # ----------------------------------------------------
        # Predictions
        # ----------------------------------------------------

        predictions = torch.argmax(
            outputs,
            dim=1
        )


        all_labels.extend(
            labels
            .detach()
            .cpu()
            .numpy()
        )

        all_predictions.extend(
            predictions
            .detach()
            .cpu()
            .numpy()
        )


    average_loss = (
        total_loss
        / len(train_dataset)
    )


    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )


    return (
        average_loss,
        accuracy
    )


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

            images = images.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )


            outputs = model(
                images
            )


            loss = criterion(
                outputs,
                labels
            )


            total_loss += (
                loss.item()
                * images.size(0)
            )


            predictions = torch.argmax(
                outputs,
                dim=1
            )


            all_labels.extend(
                labels
                .cpu()
                .numpy()
            )

            all_predictions.extend(
                predictions
                .cpu()
                .numpy()
            )


    average_loss = (
        total_loss
        / len(val_dataset)
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
        f1
    )


# ============================================================
# TRAINING LOOP
# ============================================================

best_val_f1 = -1.0

history = {
    "train_loss": [],
    "train_accuracy": [],
    "val_loss": [],
    "val_accuracy": [],
    "val_precision": [],
    "val_recall": [],
    "val_f1": []
}


print("\n")

print(
    "=" * 70
)

print(
    "STARTING TRAINING"
)

print(
    "=" * 70
)


for epoch in range(
    NUM_EPOCHS
):

    (
        train_loss,
        train_accuracy
    ) = train_one_epoch()


    (
        val_loss,
        val_accuracy,
        val_precision,
        val_recall,
        val_f1
    ) = validate()


    # --------------------------------------------------------
    # Save history
    # --------------------------------------------------------

    history[
        "train_loss"
    ].append(
        train_loss
    )

    history[
        "train_accuracy"
    ].append(
        train_accuracy
    )

    history[
        "val_loss"
    ].append(
        val_loss
    )

    history[
        "val_accuracy"
    ].append(
        val_accuracy
    )

    history[
        "val_precision"
    ].append(
        val_precision
    )

    history[
        "val_recall"
    ].append(
        val_recall
    )

    history[
        "val_f1"
    ].append(
        val_f1
    )


    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print(
        "\n" + "-" * 70
    )

    print(
        f"Epoch "
        f"{epoch + 1}/{NUM_EPOCHS}"
    )

    print(
        f"Train Loss       : "
        f"{train_loss:.4f}"
    )

    print(
        f"Train Accuracy   : "
        f"{train_accuracy:.4f}"
    )

    print(
        f"Validation Loss  : "
        f"{val_loss:.4f}"
    )

    print(
        f"Validation Acc   : "
        f"{val_accuracy:.4f}"
    )

    print(
        f"Validation Prec  : "
        f"{val_precision:.4f}"
    )

    print(
        f"Validation Recall: "
        f"{val_recall:.4f}"
    )

    print(
        f"Validation F1    : "
        f"{val_f1:.4f}"
    )


    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    if val_f1 > best_val_f1:

        best_val_f1 = val_f1


        torch.save(
            {
                "epoch":
                    epoch + 1,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "class_to_idx":
                    train_dataset.class_to_idx,

                "validation_f1":
                    best_val_f1,

                "architecture":
                    "EfficientNet-B0 + "
                    "Dense(256, ReLU) + "
                    "Dense(64, ReLU) + "
                    "Dense(2)",

                "learning_rate":
                    LEARNING_RATE,

                "batch_size":
                    BATCH_SIZE,

            },
            BEST_MODEL_PATH
        )


        print(
            "*** Best model saved "
            f"(Validation F1 = "
            f"{best_val_f1:.4f})"
        )


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\n")

print(
    "=" * 70
)

print(
    "LOADING BEST MODEL"
)

print(
    "=" * 70
)


checkpoint = torch.load(
    BEST_MODEL_PATH,
    map_location=DEVICE
)


model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)


print(
    f"Best epoch: "
    f"{checkpoint['epoch']}"
)

print(
    f"Best validation F1: "
    f"{checkpoint['validation_f1']:.4f}"
)


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

print("\n")

print(
    "=" * 70
)

print(
    "FINAL TEST EVALUATION"
)

print(
    "=" * 70
)


model.eval()

all_test_labels = []
all_test_predictions = []


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(
            DEVICE
        )


        outputs = model(
            images
        )


        predictions = torch.argmax(
            outputs,
            dim=1
        )


        all_test_labels.extend(
            labels.numpy()
        )

        all_test_predictions.extend(
            predictions
            .cpu()
            .numpy()
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


print(
    "\nTest Results:"
)


print(
    f"Accuracy  : "
    f"{test_accuracy:.4f}"
)

print(
    f"Precision : "
    f"{test_precision:.4f}"
)

print(
    f"Recall    : "
    f"{test_recall:.4f}"
)

print(
    f"F1 Score  : "
    f"{test_f1:.4f}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

class_names = [
    name
    for name, index
    in sorted(
        train_dataset.class_to_idx.items(),
        key=lambda item: item[1]
    )
]


report = classification_report(
    all_test_labels,
    all_test_predictions,
    target_names=class_names,
    zero_division=0
)


print(
    "\nClassification Report:"
)

print(
    report
)


# ============================================================
# SAVE REPORT
# ============================================================

report_path = (
    RESULTS_DIR
    / "classification_report_b0_2layers.txt"
)


with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "EfficientNet-B0 "
        "GOOD/BAD Classification\n"
    )

    file.write(
        "Experiment: "
        "B0 + 2 Extra Layers\n\n"
    )

    file.write(
        f"Learning Rate: "
        f"{LEARNING_RATE}\n"
    )

    file.write(
        f"Batch Size: "
        f"{BATCH_SIZE}\n"
    )

    file.write(
        f"Epochs: "
        f"{NUM_EPOCHS}\n"
    )

    file.write(
        "Backbone: "
        "EfficientNet-B0 frozen\n"
    )

    file.write(
        "Extra Layer 1: "
        "Linear(1280, 256) + ReLU\n"
    )

    file.write(
        "Extra Layer 2: "
        "Linear(256, 64) + ReLU\n"
    )

    file.write(
        "Classification Layer: "
        "Linear(64, 2)\n\n"
    )

    file.write(
        f"Test Accuracy  : "
        f"{test_accuracy:.4f}\n"
    )

    file.write(
        f"Test Precision : "
        f"{test_precision:.4f}\n"
    )

    file.write(
        f"Test Recall    : "
        f"{test_recall:.4f}\n"
    )

    file.write(
        f"Test F1        : "
        f"{test_f1:.4f}\n\n"
    )

    file.write(
        report
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_test_labels,
    all_test_predictions
)


print(
    "\nConfusion Matrix:"
)

print(
    cm
)


plt.figure(
    figsize=(6, 5)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.title(
    "EfficientNet-B0 "
    "GOOD/BAD - 2 Extra Layers"
)

plt.tight_layout()


cm_path = (
    RESULTS_DIR
    / "confusion_matrix_b0_2layers.png"
)


plt.savefig(
    cm_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# TRAINING CURVES
# ============================================================

epochs = range(
    1,
    NUM_EPOCHS + 1
)


plt.figure(
    figsize=(10, 6)
)

plt.plot(
    epochs,
    history["train_accuracy"],
    label="Training Accuracy"
)

plt.plot(
    epochs,
    history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy"
)

plt.title(
    "EfficientNet-B0 "
    "2 Extra Layers - Accuracy"
)

plt.legend()

plt.tight_layout()


curve_path = (
    RESULTS_DIR
    / "training_curves_b0_2layers.png"
)


plt.savefig(
    curve_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SAVE FINAL MODEL
# ============================================================

final_model_path = (
    MODEL_DIR
    / "efficientnet_b0_good_bad_2layers_final.pth"
)


torch.save(
    {
        "model_state_dict":
            model.state_dict(),

        "class_to_idx":
            train_dataset.class_to_idx,

        "architecture":
            "EfficientNet-B0 + "
            "Linear(256, ReLU) + "
            "Linear(64, ReLU) + "
            "Linear(2)"
    },
    final_model_path
)


# ============================================================
# COMPLETE
# ============================================================

print("\n")

print(
    "=" * 70
)

print(
    "TRAINING COMPLETE"
)

print(
    "=" * 70
)

print(
    "\nBest model saved to:"
)

print(
    BEST_MODEL_PATH
)

print(
    "\nFinal model saved to:"
)

print(
    final_model_path
)

print(
    "\nClassification report:"
)

print(
    report_path
)

print(
    "\nConfusion matrix:"
)

print(
    cm_path
)

print(
    "\nTraining curves:"
)

print(
    curve_path
)

print(
    "\nClass mapping:"
)

print(
    train_dataset.class_to_idx
)

print(
    "\nDone."
)
