# ============================================================
# train_lite.py
# EfficientNet-Lite0
# Binary classification: bad vs good
# ============================================================

import os

# Reduce TensorFlow informational output
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

IMG_SIZE = 224
BATCH_SIZE = 16

MAX_EPOCHS = 20
LEARNING_RATE = 1e-3

SEED = 42

# IMPORTANT:
# 0 = bad
# 1 = good
CLASS_NAMES = ["bad", "good"]


# ============================================================
# 2. PROJECT PATHS
# ============================================================
#
# Folder structure:
#
# EfficientNet_B0/
# |
# |-- Dataset_Split/
# |   |-- train/
# |   |   |-- bad/
# |   |   `-- good/
# |   |
# |   |-- validation/
# |   |   |-- bad/
# |   |   `-- good/
# |   |
# |   `-- test/
# |       |-- bad/
# |       `-- good/
# |
# |-- Models/
# |-- results/
# |
# `-- Src/
#     `-- train_lite.py
#
# ============================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

BASE_DIR = os.path.dirname(
    SCRIPT_DIR
)

DATASET_DIR = os.path.join(
    BASE_DIR,
    "Dataset_Split"
)

TRAIN_DIR = os.path.join(
    DATASET_DIR,
    "train"
)

VAL_DIR = os.path.join(
    DATASET_DIR,
    "validation"
)

TEST_DIR = os.path.join(
    DATASET_DIR,
    "test"
)

MODELS_DIR = os.path.join(
    BASE_DIR,
    "Models"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)


os.makedirs(
    MODELS_DIR,
    exist_ok=True
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# 3. OUTPUT FILES
# ============================================================

WEIGHTS_PATH = os.path.join(
    MODELS_DIR,
    "efficientnet_lite0_good_bad_lite.weights.h5"
)

FINAL_WEIGHTS_PATH = os.path.join(
    MODELS_DIR,
    "efficientnet_lite0_good_bad_lite_final.weights.h5"
)

REPORT_PATH = os.path.join(
    RESULTS_DIR,
    "classification_report_lite.txt"
)

CONFUSION_MATRIX_PATH = os.path.join(
    RESULTS_DIR,
    "confusion_matrix_lite.png"
)

TRAINING_CURVES_PATH = os.path.join(
    RESULTS_DIR,
    "training_curves_lite.png"
)


# ============================================================
# 4. EFFICIENTNET-LITE0 PRETRAINED MODEL
# ============================================================

EFFICIENTNET_LITE_URL = (
    "https://tfhub.dev/tensorflow/"
    "efficientnet/lite0/feature-vector/2"
)


# ============================================================
# 5. REPRODUCIBILITY
# ============================================================

tf.keras.utils.set_random_seed(
    SEED
)

np.random.seed(
    SEED
)


# ============================================================
# 6. SYSTEM INFORMATION
# ============================================================

print()
print("=" * 70)
print("EFFICIENTNET-LITE0 TRAINING")
print("=" * 70)

print(
    "TensorFlow version:",
    tf.__version__
)

gpus = tf.config.list_physical_devices(
    "GPU"
)

if gpus:

    print("GPU detected:")

    for gpu in gpus:
        print(" ", gpu)

else:

    print(
        "No GPU detected. "
        "Training will use CPU."
    )

print()


# ============================================================
# 7. VERIFY DATASET DIRECTORIES
# ============================================================

for directory in [
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR,
]:

    if not os.path.isdir(directory):

        raise FileNotFoundError(
            "\nDataset directory was not found:\n"
            f"{directory}\n"
        )


# ============================================================
# 8. LOAD TRAINING DATASET
# ============================================================

print("=" * 70)
print("LOADING DATASETS")
print("=" * 70)

train_ds = (
    tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        labels="inferred",
        label_mode="binary",
        class_names=CLASS_NAMES,
        image_size=(
            IMG_SIZE,
            IMG_SIZE
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED,
    )
)


# ============================================================
# 9. LOAD VALIDATION DATASET
# ============================================================

val_ds = (
    tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        labels="inferred",
        label_mode="binary",
        class_names=CLASS_NAMES,
        image_size=(
            IMG_SIZE,
            IMG_SIZE
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
)


# ============================================================
# 10. LOAD TEST DATASET
# ============================================================

test_ds = (
    tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        labels="inferred",
        label_mode="binary",
        class_names=CLASS_NAMES,
        image_size=(
            IMG_SIZE,
            IMG_SIZE
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
)


print()
print(
    "Classes:",
    train_ds.class_names
)

print()


# ============================================================
# 11. DATASET PERFORMANCE
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(
    AUTOTUNE
)

val_ds = val_ds.prefetch(
    AUTOTUNE
)

test_ds = test_ds.prefetch(
    AUTOTUNE
)


# ============================================================
# 12. DATA AUGMENTATION
# ============================================================
#
# Moderate augmentation because the current dataset is small.
#
# No horizontal/vertical flip is added because we do not know
# whether orientation is semantically safe for your images.
# ============================================================

data_augmentation = tf.keras.Sequential(
    [

        tf.keras.layers.RandomRotation(
            factor=0.05
        ),

        tf.keras.layers.RandomZoom(
            height_factor=0.10,
            width_factor=0.10
        ),

        tf.keras.layers.RandomContrast(
            factor=0.10
        ),

    ],
    name="data_augmentation"
)


# ============================================================
# 13. DOWNLOAD / RESOLVE EFFICIENTNET-LITE0
# ============================================================

print("=" * 70)
print("LOADING PRETRAINED EFFICIENTNET-LITE0")
print("=" * 70)

lite_path = hub.resolve(
    EFFICIENTNET_LITE_URL
)

print()
print("EfficientNet-Lite0 path:")
print(lite_path)
print()


# ============================================================
# 14. LOAD THE CORRECT SAVEDMODEL METAGRAPH
# ============================================================
#
# Your TF Hub model contains multiple MetaGraphs.
#
# Your previous output showed:
#
# ['train']
# []
#
# We want the default inference graph here.
# ============================================================

lite_module = tf.saved_model.load(
    lite_path,
    tags=[]
)

print(
    "EfficientNet-Lite0 "
    "SavedModel loaded successfully."
)

print()


# ============================================================
# 15. DISPLAY AVAILABLE SIGNATURES
# ============================================================

print("Available signatures:")

for signature_name in lite_module.signatures:

    print(
        "  ",
        signature_name
    )

print()


# ============================================================
# 16. SELECT FEATURE EXTRACTION SIGNATURE
# ============================================================
#
# Your actual model exposed:
#
# default
# image_feature_vector
#
# Prefer image_feature_vector.
# ============================================================

if (
    "image_feature_vector"
    in lite_module.signatures
):

    lite_signature = (
        lite_module.signatures[
            "image_feature_vector"
        ]
    )

    selected_signature_name = (
        "image_feature_vector"
    )

elif (
    "default"
    in lite_module.signatures
):

    lite_signature = (
        lite_module.signatures[
            "default"
        ]
    )

    selected_signature_name = (
        "default"
    )

else:

    raise RuntimeError(
        "\nNo usable EfficientNet-Lite0 "
        "feature extraction signature "
        "was found.\n"
    )


print(
    "Selected signature:",
    selected_signature_name
)

print()


# ============================================================
# 17. SELECT CORRECT OUTPUT
# ============================================================
#
# IMPORTANT FIX
#
# DO NOT do:
#
# list(outputs.keys())[0]
#
# Your previous run did that and selected:
#
# block_12
#
# block_12 shape:
#
# (None, 7, 7, 192)
#
# That resulted in:
#
# Dense -> (None, 7, 7, 1)
#
# But labels are:
#
# (None, 1)
#
# We specifically need the pooled 1280-dimensional vector:
#
# default         -> (None, 1280)
# pooled_features -> (None, 1280)
#
# ============================================================

signature_outputs = (
    lite_signature.structured_outputs
)


if "default" in signature_outputs:

    OUTPUT_KEY = "default"

elif "pooled_features" in signature_outputs:

    OUTPUT_KEY = "pooled_features"

else:

    raise RuntimeError(
        "\nEfficientNet-Lite0 does not expose "
        "'default' or 'pooled_features'.\n"
        f"Available outputs:\n"
        f"{list(signature_outputs.keys())}\n"
    )


feature_shape = (
    signature_outputs[
        OUTPUT_KEY
    ].shape
)


print(
    "Using output key:",
    OUTPUT_KEY
)

print(
    "Feature vector shape:",
    feature_shape
)

print()


# ============================================================
# 18. SAFETY CHECK
# ============================================================
#
# We expect:
#
# (None, 1280)
#
# NOT:
#
# (None, 7, 7, 192)
#
# ============================================================

if len(feature_shape) != 2:

    raise RuntimeError(
        "\nWrong EfficientNet-Lite output "
        "was selected.\n\n"
        f"Selected output: {OUTPUT_KEY}\n"
        f"Selected shape: {feature_shape}\n\n"
        "Expected a 2D pooled feature vector "
        "such as (None, 1280).\n"
    )


# ============================================================
# 19. CUSTOM FROZEN FEATURE EXTRACTOR
# ============================================================

class EfficientNetLiteFeatureExtractor(
    tf.keras.layers.Layer
):

    def __init__(
        self,
        signature,
        output_key,
        **kwargs
    ):

        super().__init__(
            trainable=False,
            **kwargs
        )

        self.signature = signature

        self.output_key = output_key


    def call(
        self,
        images
    ):

        outputs = self.signature(
            images
        )

        features = outputs[
            self.output_key
        ]

        return features


# ============================================================
# 20. CREATE FEATURE EXTRACTOR
# ============================================================

feature_extractor = (
    EfficientNetLiteFeatureExtractor(
        signature=lite_signature,
        output_key=OUTPUT_KEY,
        name="efficientnet_lite0_features"
    )
)


# ============================================================
# 21. BUILD CLASSIFICATION MODEL
# ============================================================

print("=" * 70)
print("BUILDING CLASSIFIER")
print("=" * 70)

inputs = tf.keras.Input(
    shape=(
        IMG_SIZE,
        IMG_SIZE,
        3
    ),
    dtype=tf.float32,
    name="input_image"
)


# Data augmentation is active only during training
x = data_augmentation(
    inputs
)


# Pretrained EfficientNet-Lite0
#
# Expected:
#
# (batch, 224, 224, 3)
#       ->
# (batch, 1280)

x = feature_extractor(
    x
)


# Dropout
x = tf.keras.layers.Dropout(
    rate=0.30,
    name="dropout"
)(x)


# Binary classifier
#
# Output:
#
# (batch, 1)

outputs = tf.keras.layers.Dense(
    units=1,
    activation="sigmoid",
    name="classifier"
)(x)


model = tf.keras.Model(
    inputs=inputs,
    outputs=outputs,
    name="EfficientNet_Lite0_Good_Bad"
)


# ============================================================
# 22. VERIFY MODEL OUTPUT SHAPE
# ============================================================

print()
print(
    "Final model output shape:",
    model.output_shape
)

if model.output_shape[-1] != 1:

    raise RuntimeError(
        "\nClassifier output is incorrect.\n"
        f"Received: {model.output_shape}\n"
        "Expected: (None, 1)\n"
    )


# ============================================================
# 23. COMPILE MODEL
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),

    loss=tf.keras.losses.BinaryCrossentropy(),

    metrics=[

        tf.keras.metrics.BinaryAccuracy(
            name="accuracy"
        ),

        tf.keras.metrics.Precision(
            name="precision"
        ),

        tf.keras.metrics.Recall(
            name="recall"
        ),

    ]
)


# ============================================================
# 24. MODEL SUMMARY
# ============================================================

print()
print("=" * 70)
print("MODEL SUMMARY")
print("=" * 70)

model.summary()

print()


# ============================================================
# 25. IMPORTANT ARCHITECTURE CHECK
# ============================================================

print("=" * 70)
print("ARCHITECTURE CHECK")
print("=" * 70)

print(
    "Feature extractor output:",
    feature_shape
)

print(
    "Classifier output:",
    model.output_shape
)

print()

print(
    "Expected feature extractor output: "
    "(None, 1280)"
)

print(
    "Expected classifier output: "
    "(None, 1)"
)

print()


# ============================================================
# 26. CALLBACKS
# ============================================================

checkpoint = (
    tf.keras.callbacks.ModelCheckpoint(

        filepath=WEIGHTS_PATH,

        monitor="val_loss",

        mode="min",

        save_best_only=True,

        save_weights_only=True,

        verbose=1
    )
)


early_stopping = (
    tf.keras.callbacks.EarlyStopping(

        monitor="val_loss",

        mode="min",

        patience=5,

        restore_best_weights=True,

        verbose=1
    )
)


reduce_lr = (
    tf.keras.callbacks.ReduceLROnPlateau(

        monitor="val_loss",

        mode="min",

        factor=0.5,

        patience=2,

        min_lr=1e-6,

        verbose=1
    )
)


# ============================================================
# 27. TRAIN
# ============================================================

print("=" * 70)
print("TRAINING")
print("=" * 70)

print(
    "Learning rate:",
    LEARNING_RATE
)

print(
    "Batch size:",
    BATCH_SIZE
)

print(
    "Maximum epochs:",
    MAX_EPOCHS
)

print()


history = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=MAX_EPOCHS,

    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr,
    ],

    verbose=1
)


# ============================================================
# 28. LOAD BEST WEIGHTS
# ============================================================

print()
print("=" * 70)
print("LOADING BEST MODEL WEIGHTS")
print("=" * 70)


if os.path.exists(
    WEIGHTS_PATH
):

    model.load_weights(
        WEIGHTS_PATH
    )

    print(
        "Best validation weights loaded."
    )

else:

    print(
        "Checkpoint was not found. "
        "Using current model weights."
    )


# ============================================================
# 29. TEST EVALUATION
# ============================================================

print()
print("=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)


test_results = model.evaluate(
    test_ds,
    verbose=1,
    return_dict=True
)


print()

for metric_name, metric_value in (
    test_results.items()
):

    print(
        f"{metric_name}: "
        f"{metric_value:.4f}"
    )


# ============================================================
# 30. TEST PREDICTIONS
# ============================================================

print()
print("=" * 70)
print("GENERATING TEST PREDICTIONS")
print("=" * 70)


y_true = []
y_probability = []


for images, labels in test_ds:

    probabilities = model(
        images,
        training=False
    ).numpy()

    y_probability.extend(
        probabilities.reshape(-1)
    )

    y_true.extend(
        labels.numpy().reshape(-1)
    )


y_true = np.asarray(
    y_true,
    dtype=np.int32
)

y_probability = np.asarray(
    y_probability,
    dtype=np.float32
)


# ============================================================
# 31. BINARY CLASSIFICATION
# ============================================================
#
# Folder/class mapping:
#
# bad  = 0
# good = 1
#
# Therefore:
#
# probability < 0.5  -> bad
# probability >= 0.5 -> good
#
# ============================================================

y_pred = (
    y_probability >= 0.5
).astype(
    np.int32
)


# ============================================================
# 32. CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0
)


print()
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(report)


# ============================================================
# 33. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[
        0,
        1
    ]
)


print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(cm)

print()


# ============================================================
# 34. SAVE CLASSIFICATION REPORT
# ============================================================

with open(
    REPORT_PATH,
    "w",
    encoding="utf-8"
) as report_file:

    report_file.write(
        "EfficientNet-Lite0 "
        "Good vs Bad Classification\n"
    )

    report_file.write(
        "=" * 70
    )

    report_file.write(
        "\n\n"
    )

    report_file.write(
        f"TensorFlow version: "
        f"{tf.__version__}\n"
    )

    report_file.write(
        "Model: EfficientNet-Lite0\n"
    )

    report_file.write(
        f"Pretrained source: "
        f"{EFFICIENTNET_LITE_URL}\n"
    )

    report_file.write(
        f"Image size: "
        f"{IMG_SIZE}x{IMG_SIZE}\n"
    )

    report_file.write(
        f"Batch size: "
        f"{BATCH_SIZE}\n"
    )

    report_file.write(
        f"Initial learning rate: "
        f"{LEARNING_RATE}\n"
    )

    report_file.write(
        f"Maximum epochs: "
        f"{MAX_EPOCHS}\n"
    )

    report_file.write(
        f"Output key: "
        f"{OUTPUT_KEY}\n"
    )

    report_file.write(
        f"Feature shape: "
        f"{feature_shape}\n"
    )

    report_file.write(
        f"Test images: "
        f"{len(y_true)}\n"
    )

    report_file.write(
        "\nTest metrics:\n"
    )

    for metric_name, metric_value in (
        test_results.items()
    ):

        report_file.write(
            f"{metric_name}: "
            f"{metric_value:.4f}\n"
        )

    report_file.write(
        "\nClassification report:\n\n"
    )

    report_file.write(
        report
    )

    report_file.write(
        "\nConfusion matrix:\n"
    )

    report_file.write(
        np.array2string(
            cm
        )
    )

    report_file.write(
        "\n"
    )


# ============================================================
# 35. SAVE CONFUSION MATRIX IMAGE
# ============================================================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES
)

fig, ax = plt.subplots(
    figsize=(
        6,
        6
    )
)

disp.plot(
    ax=ax,
    values_format="d"
)

ax.set_title(
    "EfficientNet-Lite0 "
    "Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    CONFUSION_MATRIX_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close(
    fig
)


# ============================================================
# 36. SAVE TRAINING CURVES
# ============================================================

fig = plt.figure(
    figsize=(
        10,
        6
    )
)


plt.plot(
    history.history[
        "accuracy"
    ],
    label="Training Accuracy"
)


plt.plot(
    history.history[
        "val_accuracy"
    ],
    label="Validation Accuracy"
)


plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Accuracy"
)

plt.title(
    "EfficientNet-Lite0 "
    "Training and Validation Accuracy"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    TRAINING_CURVES_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close(
    fig
)


# ============================================================
# 37. SAVE FINAL WEIGHTS
# ============================================================

model.save_weights(
    FINAL_WEIGHTS_PATH
)


# ============================================================
# 38. FINISHED
# ============================================================

print()
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print()

print(
    "Best model weights:"
)

print(
    WEIGHTS_PATH
)

print()

print(
    "Final model weights:"
)

print(
    FINAL_WEIGHTS_PATH
)

print()

print(
    "Classification report:"
)

print(
    REPORT_PATH
)

print()

print(
    "Confusion matrix:"
)

print(
    CONFUSION_MATRIX_PATH
)

print()

print(
    "Training curves:"
)

print(
    TRAINING_CURVES_PATH
)

print()

print(
    "Class mapping:"
)

print(
    "0 = bad"
)

print(
    "1 = good"
)

print()

print(
    "Done."
)