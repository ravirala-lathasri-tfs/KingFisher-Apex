"""Train a frozen EfficientNet-Lite0 backbone on the three-class split."""
import argparse
import csv
import json
import random
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import timm
from timm.data import resolve_model_data_config, create_transform

DEFAULT_DATA = Path(r"C:\Users\lathasri.ravirala\OneDrive - Thermo Fisher Scientific\KF-Barcode\Dataset_3_classes\train_test_validation_split")


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.eval()  # Keep frozen backbone batch-normalization statistics fixed.
    model.classifier.train(training)
    loss_sum = correct = count = gradient_sum = batches = 0
    confusion = torch.zeros((len(loader.dataset.classes),) * 2, dtype=torch.int64)
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            if training:
                loss.backward()
                gradient_sum += math.sqrt(sum(p.grad.detach().float().pow(2).sum().item() for p in model.classifier.parameters() if p.grad is not None))
                batches += 1
                optimizer.step()
            predictions = outputs.argmax(dim=1)
            count += labels.size(0)
            loss_sum += loss.item() * labels.size(0)
            correct += (predictions == labels).sum().item()
            for actual, predicted in zip(labels.cpu().tolist(), predictions.cpu().tolist()):
                confusion[actual, predicted] += 1
    return loss_sum / count, correct / count, confusion.tolist(), (gradient_sum / batches if batches else None)


def save_training_plots(history, output_dir):
    epochs, train_loss, train_acc, val_loss, val_acc, gradients = zip(*history)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, train_loss, label='Training')
    axes[0].plot(epochs, val_loss, label='Validation')
    axes[0].set(xlabel='Epoch', ylabel='Cross-entropy loss', title='Loss')
    axes[1].plot(epochs, train_acc, label='Training')
    axes[1].plot(epochs, val_acc, label='Validation')
    axes[1].set(xlabel='Epoch', ylabel='Accuracy', title='Accuracy', ylim=(0, 1))
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / 'training_curves.png', dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, val_acc, marker='o')
    ax.set(xlabel='Epoch', ylabel='Validation accuracy', title='Validation accuracy by epoch', ylim=(0, 1))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / 'validation_accuracy.png', dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, gradients, marker='o')
    ax.set(xlabel='Epoch', ylabel='Mean gradient L2 norm', title='Training gradient magnitude by epoch')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / 'gradient_norm.png', dpi=160)
    plt.close(fig)


def save_confusion_matrix(matrix, classes, output_dir):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap='Blues')
    fig.colorbar(im, ax=ax, label='Images')
    ax.set(xticks=range(len(classes)), yticks=range(len(classes)), xticklabels=classes,
           yticklabels=classes, xlabel='Predicted class', ylabel='Actual class', title='Test confusion matrix')
    maximum = max(max(row) for row in matrix)
    for row, values in enumerate(matrix):
        for col, value in enumerate(values):
            ax.text(col, row, str(value), ha='center', va='center',
                    color='white' if value > maximum / 2 else 'black')
    fig.tight_layout()
    fig.savefig(output_dir / 'confusion_matrix.png', dpi=160)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'results' / 'lite0_three_class')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch_size) <= 0 or args.learning_rate <= 0 or args.workers < 0:
        parser.error('epochs, patience, batch size, and learning rate must be positive; workers must be nonnegative')
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}', flush=True)
    print('Loading ImageNet-pretrained EfficientNet-Lite0 (weights may be downloaded)...', flush=True)
    model = timm.create_model('tf_efficientnet_lite0', pretrained=True)
    config = resolve_model_data_config(model)
    transform = create_transform(**config, is_training=False)
    datasets = {name: ImageFolder(args.data_dir / name, transform=transform) for name in ('train', 'validation', 'test')}
    classes = datasets['train'].classes
    if set(classes) != {'good', 'bad', 'average'} or any(ds.classes != classes for ds in datasets.values()):
        raise ValueError('All splits must contain exactly the same good, bad, and average class folders.')
    loaders = {name: DataLoader(ds, batch_size=args.batch_size, shuffle=(name == 'train'), num_workers=args.workers, pin_memory=(device.type == 'cuda')) for name, ds in datasets.items()}
    for parameter in model.parameters():
        parameter.requires_grad = False
    features = model.get_classifier().in_features
    model.classifier = nn.Sequential(nn.Linear(features, 512), nn.ReLU(), nn.Dropout(0.2), nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, len(classes)))
    model.to(device)
    optimizer = torch.optim.SGD(model.classifier.parameters(), lr=args.learning_rate, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(__file__).resolve().parent / 'Models'
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = model_dir / 'efficientnet_lite0_three_class_best.pt'
    history_path = args.output_dir / 'history.csv'
    history = []
    best_loss = float('inf')
    stale = 0
    print(f'Classes: {classes}; counts: ' + ', '.join(f'{k}={len(v)}' for k, v in datasets.items()), flush=True)
    with history_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['epoch', 'train_loss', 'train_accuracy', 'validation_loss', 'validation_accuracy', 'mean_gradient_norm'])
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc, _, gradient_norm = run_epoch(model, loaders['train'], criterion, device, optimizer)
            val_loss, val_acc, _, _ = run_epoch(model, loaders['validation'], criterion, device)
            writer.writerow([epoch, train_loss, train_acc, val_loss, val_acc, gradient_norm])
            history.append((epoch, train_loss, train_acc, val_loss, val_acc, gradient_norm))
            save_training_plots(history, args.output_dir)
            handle.flush()
            print(f'Epoch {epoch}/{args.epochs}: train loss={train_loss:.4f} acc={train_acc:.4f}; validation loss={val_loss:.4f} acc={val_acc:.4f}', flush=True)
            if val_loss < best_loss:
                best_loss, stale = val_loss, 0
                torch.save({'model_state_dict': model.state_dict(), 'classes': classes, 'model_name': 'tf_efficientnet_lite0', 'preprocessing': config, 'epoch': epoch, 'validation_loss': val_loss}, checkpoint)
            else:
                stale += 1
                if stale >= args.patience:
                    print('Early stopping: validation loss did not improve.', flush=True)
                    break
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(saved['model_state_dict'])
    test_loss, test_acc, confusion, _ = run_epoch(model, loaders['test'], criterion, device)
    save_confusion_matrix(confusion, classes, args.output_dir)
    results = {'classes': classes, 'best_epoch': saved['epoch'], 'best_validation_loss': best_loss, 'test_loss': test_loss, 'test_accuracy': test_acc, 'confusion_matrix_rows_actual_columns_predicted': confusion}
    (args.output_dir / 'test_results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(f'Test loss={test_loss:.4f}, accuracy={test_acc:.4f}; best model: {checkpoint}', flush=True)
    print('Confusion matrix (rows=actual, columns=predicted):', confusion, flush=True)


if __name__ == '__main__':
    main()


