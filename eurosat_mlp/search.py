import numpy as np
import os
import json
import itertools
import time

from model import ThreeLayerMLP, cross_entropy_loss
from optimizer import SGD, StepLRScheduler
from data_loader import prepare_data, augment_batch
from train import compute_accuracy, compute_loss


def grid_search(data_dir, save_dir='search_results', seed=42):
    os.makedirs(save_dir, exist_ok=True)

    data = prepare_data(data_dir, seed=seed)
    train_X, train_y = data['train_X'], data['train_y']
    val_X, val_y = data['val_X'], data['val_y']
    input_dim = train_X.shape[1]
    num_classes = 10

    param_grid = {
        'lr': [0.01, 0.005, 0.001],
        'hidden_dim': [128, 256, 512],
        'weight_decay': [0, 1e-4, 1e-3],
    }

    search_epochs = 20
    batch_size = 128

    all_combos = list(itertools.product(
        param_grid['lr'],
        param_grid['hidden_dim'],
        param_grid['weight_decay'],
    ))

    results = []
    best_val_acc = 0.0
    best_config = None

    print(f"Grid search: {len(all_combos)} combinations, {search_epochs} epochs each\n")

    for i, (lr, hidden_dim, wd) in enumerate(all_combos):
        config = {
            'lr': lr,
            'hidden1_dim': hidden_dim,
            'hidden2_dim': hidden_dim // 2,
            'weight_decay': wd,
        }

        print(f"[{i + 1}/{len(all_combos)}] lr={lr}, hidden={hidden_dim}, wd={wd}")
        t0 = time.time()

        np.random.seed(seed)
        model = ThreeLayerMLP(
            input_dim, config['hidden1_dim'], config['hidden2_dim'],
            num_classes, activation='relu'
        )
        optimizer = SGD(lr=lr)
        scheduler = StepLRScheduler(optimizer, step_size=8, gamma=0.5)

        trial_best_val_acc = 0.0

        for epoch in range(1, search_epochs + 1):
            scheduler.step(epoch - 1)
            indices = np.random.permutation(train_X.shape[0])

            for start in range(0, train_X.shape[0], batch_size):
                end = min(start + batch_size, train_X.shape[0])
                batch_idx = indices[start:end]
                X_b = augment_batch(train_X[batch_idx])
                y_b = train_y[batch_idx]

                model.forward(X_b)
                model.backward(y_b, weight_decay=wd)
                optimizer.step(model)

            val_acc = compute_accuracy(model, val_X, val_y)
            if val_acc > trial_best_val_acc:
                trial_best_val_acc = val_acc

        elapsed = time.time() - t0
        train_acc = compute_accuracy(model, train_X, train_y)

        result_entry = {
            **config,
            'best_val_acc': float(trial_best_val_acc),
            'final_train_acc': float(train_acc),
            'time': float(elapsed),
        }
        results.append(result_entry)

        print(f"  => val_acc={trial_best_val_acc:.4f}, train_acc={train_acc:.4f}, time={elapsed:.1f}s")

        if trial_best_val_acc > best_val_acc:
            best_val_acc = trial_best_val_acc
            best_config = config.copy()
            model.save(os.path.join(save_dir, 'search_best_model.npz'))

    results_sorted = sorted(results, key=lambda x: x['best_val_acc'], reverse=True)

    search_report = {
        'param_grid': param_grid,
        'search_epochs': search_epochs,
        'results': results_sorted,
        'best_config': best_config,
        'best_val_acc': float(best_val_acc),
    }

    with open(os.path.join(save_dir, 'search_results.json'), 'w') as f:
        json.dump(search_report, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Grid Search Complete!")
    print(f"Best val accuracy: {best_val_acc:.4f}")
    print(f"Best config: {best_config}")
    print(f"\nTop 5 configurations:")
    for i, r in enumerate(results_sorted[:5]):
        print(f"  {i + 1}. val_acc={r['best_val_acc']:.4f} | "
              f"lr={r['lr']}, hidden={r['hidden1_dim']}, wd={r['weight_decay']}")
    print(f"\nResults saved to {save_dir}/search_results.json")

    return search_report


def random_search(data_dir, n_trials=20, save_dir='search_results', seed=42):
    os.makedirs(save_dir, exist_ok=True)

    data = prepare_data(data_dir, seed=seed)
    train_X, train_y = data['train_X'], data['train_y']
    val_X, val_y = data['val_X'], data['val_y']
    input_dim = train_X.shape[1]
    num_classes = 10

    rng = np.random.RandomState(seed)
    search_epochs = 20
    batch_size = 128

    results = []
    best_val_acc = 0.0
    best_config = None

    print(f"Random search: {n_trials} trials, {search_epochs} epochs each\n")

    for trial in range(1, n_trials + 1):
        lr = 10 ** rng.uniform(-3, -1.5)
        hidden1 = rng.choice([64, 128, 256, 512, 1024])
        hidden2 = rng.choice([64, 128, 256, 512])
        wd = 10 ** rng.uniform(-6, -2)
        activation = rng.choice(['relu', 'tanh'])

        config = {
            'lr': float(lr),
            'hidden1_dim': int(hidden1),
            'hidden2_dim': int(hidden2),
            'weight_decay': float(wd),
            'activation': activation,
        }

        print(f"[{trial}/{n_trials}] lr={lr:.5f}, h1={hidden1}, h2={hidden2}, "
              f"wd={wd:.6f}, act={activation}")
        t0 = time.time()

        np.random.seed(seed + trial)
        model = ThreeLayerMLP(
            input_dim, config['hidden1_dim'], config['hidden2_dim'],
            num_classes, activation=activation
        )
        optimizer = SGD(lr=lr)
        scheduler = StepLRScheduler(optimizer, step_size=8, gamma=0.5)

        trial_best_val_acc = 0.0

        for epoch in range(1, search_epochs + 1):
            scheduler.step(epoch - 1)
            indices = np.random.permutation(train_X.shape[0])

            for start in range(0, train_X.shape[0], batch_size):
                end = min(start + batch_size, train_X.shape[0])
                batch_idx = indices[start:end]
                X_b = augment_batch(train_X[batch_idx])
                y_b = train_y[batch_idx]

                model.forward(X_b)
                model.backward(y_b, weight_decay=wd)
                optimizer.step(model)

            val_acc = compute_accuracy(model, val_X, val_y)
            if val_acc > trial_best_val_acc:
                trial_best_val_acc = val_acc

        elapsed = time.time() - t0
        train_acc = compute_accuracy(model, train_X, train_y)

        result_entry = {
            **config,
            'best_val_acc': float(trial_best_val_acc),
            'final_train_acc': float(train_acc),
            'time': float(elapsed),
        }
        results.append(result_entry)

        print(f"  => val_acc={trial_best_val_acc:.4f}, train_acc={train_acc:.4f}, time={elapsed:.1f}s")

        if trial_best_val_acc > best_val_acc:
            best_val_acc = trial_best_val_acc
            best_config = config.copy()
            model.save(os.path.join(save_dir, 'search_best_model.npz'))

    results_sorted = sorted(results, key=lambda x: x['best_val_acc'], reverse=True)

    search_report = {
        'method': 'random_search',
        'n_trials': n_trials,
        'search_epochs': search_epochs,
        'results': results_sorted,
        'best_config': best_config,
        'best_val_acc': float(best_val_acc),
    }

    with open(os.path.join(save_dir, 'random_search_results.json'), 'w') as f:
        json.dump(search_report, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Random Search Complete!")
    print(f"Best val accuracy: {best_val_acc:.4f}")
    print(f"Best config: {best_config}")
    print(f"\nTop 5 configurations:")
    for i, r in enumerate(results_sorted[:5]):
        print(f"  {i + 1}. val_acc={r['best_val_acc']:.4f} | "
              f"lr={r['lr']:.5f}, h1={r['hidden1_dim']}, h2={r['hidden2_dim']}, "
              f"wd={r['weight_decay']:.6f}, act={r['activation']}")

    return search_report


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../EuroSAT_RGB')
    parser.add_argument('--save_dir', type=str, default='search_results')
    parser.add_argument('--method', type=str, default='grid', choices=['grid', 'random'])
    parser.add_argument('--n_trials', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.method == 'grid':
        grid_search(args.data_dir, args.save_dir, args.seed)
    else:
        random_search(args.data_dir, args.n_trials, args.save_dir, args.seed)