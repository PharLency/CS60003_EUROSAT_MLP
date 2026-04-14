import numpy as np
import os
import json
import time

from model import ThreeLayerMLP, cross_entropy_loss
from optimizer import SGD, StepLRScheduler
from data_loader import prepare_data, augment_batch


def compute_accuracy(model, X, y, batch_size=512):
    n = X.shape[0]
    correct = 0
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        preds = model.predict(X[start:end])
        correct += (preds == y[start:end]).sum()
    return correct / n


def compute_loss(model, X, y, weight_decay=0.0, batch_size=512):
    n = X.shape[0]
    total_loss = 0.0
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        probs = model.forward(X[start:end])
        batch_loss = cross_entropy_loss(probs, y[start:end])
        total_loss += batch_loss * (end - start)
    ce_loss = total_loss / n

    if weight_decay > 0:
        l2_reg = 0.5 * weight_decay * (
            np.sum(model.W1 ** 2) + np.sum(model.W2 ** 2) + np.sum(model.W3 ** 2)
        )
        return ce_loss + l2_reg
    return ce_loss


def train(
    data_dir,
    save_dir='checkpoints',
    hidden1_dim=256,
    hidden2_dim=128,
    activation='relu',
    lr=0.01,
    weight_decay=1e-4,
    batch_size=128,
    epochs=50,
    lr_step_size=15,
    lr_gamma=0.5,
    seed=42,
    image_size=64,
    augment=True,
):
    np.random.seed(seed)
    os.makedirs(save_dir, exist_ok=True)

    data = prepare_data(data_dir, image_size=image_size, seed=seed)
    train_X, train_y = data['train_X'], data['train_y']
    val_X, val_y = data['val_X'], data['val_y']

    input_dim = train_X.shape[1]
    num_classes = 10

    model = ThreeLayerMLP(input_dim, hidden1_dim, hidden2_dim, num_classes, activation)
    optimizer = SGD(lr=lr)
    scheduler = StepLRScheduler(optimizer, step_size=lr_step_size, gamma=lr_gamma)

    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'lr': [],
    }

    best_val_acc = 0.0
    best_epoch = -1

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        current_lr = scheduler.step(epoch - 1)

        indices = np.random.permutation(train_X.shape[0])
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, train_X.shape[0], batch_size):
            end = min(start + batch_size, train_X.shape[0])
            batch_idx = indices[start:end]
            X_batch = train_X[batch_idx]
            y_batch = train_y[batch_idx]

            if augment:
                X_batch = augment_batch(X_batch, image_size=image_size)

            probs = model.forward(X_batch)
            batch_loss = cross_entropy_loss(probs, y_batch)
            model.backward(y_batch, weight_decay=weight_decay)
            optimizer.step(model)

            epoch_loss += batch_loss
            n_batches += 1

        train_loss = compute_loss(model, train_X, train_y, weight_decay)
        val_loss = compute_loss(model, val_X, val_y, weight_decay)
        train_acc = compute_accuracy(model, train_X, train_y)
        val_acc = compute_accuracy(model, val_X, val_y)

        history['train_loss'].append(float(train_loss))
        history['val_loss'].append(float(val_loss))
        history['train_acc'].append(float(train_acc))
        history['val_acc'].append(float(val_acc))
        history['lr'].append(float(current_lr))

        elapsed = time.time() - t0

        print(f"Epoch {epoch:3d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | "
              f"LR: {current_lr:.6f} | Time: {elapsed:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            model.save(os.path.join(save_dir, 'best_model.npz'))
            print(f"  => Best model saved (val_acc={val_acc:.4f})")

    model.save(os.path.join(save_dir, 'final_model.npz'))

    history['best_val_acc'] = float(best_val_acc)
    history['best_epoch'] = best_epoch
    history['config'] = {
        'hidden1_dim': hidden1_dim,
        'hidden2_dim': hidden2_dim,
        'activation': activation,
        'lr': lr,
        'weight_decay': weight_decay,
        'batch_size': batch_size,
        'epochs': epochs,
        'lr_step_size': lr_step_size,
        'lr_gamma': lr_gamma,
        'seed': seed,
        'augment': augment,
    }

    with open(os.path.join(save_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f} at epoch {best_epoch}")

    np.savez(os.path.join(save_dir, 'data_stats.npz'), mean=data['mean'], std=data['std'])

    return model, history, data


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../EuroSAT_RGB')
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--hidden1_dim', type=int, default=256)
    parser.add_argument('--hidden2_dim', type=int, default=128)
    parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'sigmoid', 'tanh'])
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr_step_size', type=int, default=15)
    parser.add_argument('--lr_gamma', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--augment', action='store_true', default=True)
    parser.add_argument('--no_augment', dest='augment', action='store_false')
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        hidden1_dim=args.hidden1_dim,
        hidden2_dim=args.hidden2_dim,
        activation=args.activation,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
        seed=args.seed,
        augment=args.augment,
    )