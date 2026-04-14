import numpy as np
import os
import sys
from PIL import Image
from multiprocessing import Pool, cpu_count


CLASS_NAMES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"
]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def _load_single_image(args):
    fpath, label, image_size = args
    img = Image.open(fpath).convert('RGB')
    if img.size != (image_size, image_size):
        img = img.resize((image_size, image_size))
    return np.array(img, dtype=np.float64), label, fpath


def load_dataset(data_dir, image_size=64, num_workers=None):
    cache_path = os.path.join(data_dir, f'_cache_{image_size}.npz')
    if os.path.exists(cache_path):
        print(f"Loading from cache: {cache_path}")
        cached = np.load(cache_path, allow_pickle=True)
        return cached['images'], cached['labels'], list(cached['paths'])

    all_files = []
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in sorted(os.listdir(class_dir)):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            all_files.append((os.path.join(class_dir, fname), CLASS_TO_IDX[class_name], image_size))

    total = len(all_files)
    if num_workers is None:
        num_workers = min(cpu_count(), 8)

    print(f"  Loading {total} images with {num_workers} workers...")

    with Pool(processes=num_workers) as pool:
        results = pool.map(_load_single_image, all_files, chunksize=256)

    images = np.empty((total, image_size, image_size, 3), dtype=np.float64)
    labels = np.empty(total, dtype=np.int64)
    paths = []
    for i, (img_arr, label, fpath) in enumerate(results):
        images[i] = img_arr
        labels[i] = label
        paths.append(fpath)

    print(f"Saving cache to {cache_path} ...")
    np.savez(cache_path, images=images, labels=labels, paths=np.array(paths))
    print("Cache saved.")

    return images, labels, paths


def normalize(images, mean=None, std=None):
    images = images / 255.0
    if mean is None:
        mean = images.mean(axis=(0, 1, 2), keepdims=True)
    if std is None:
        std = images.std(axis=(0, 1, 2), keepdims=True) + 1e-8
    images = (images - mean) / std
    return images, mean, std


def flatten(images):
    return images.reshape(images.shape[0], -1)


def augment_batch(X_flat, image_size=64, p_flip_h=0.5, p_flip_v=0.5, p_rot90=0.5, brightness_std=0.1):
    n = X_flat.shape[0]
    imgs = X_flat.reshape(n, image_size, image_size, 3).copy()

    for i in range(n):
        if np.random.rand() < p_flip_h:
            imgs[i] = imgs[i, :, ::-1, :]

        if np.random.rand() < p_flip_v:
            imgs[i] = imgs[i, ::-1, :, :]

        if np.random.rand() < p_rot90:
            k = np.random.choice([1, 2, 3])
            imgs[i] = np.rot90(imgs[i], k=k, axes=(0, 1))

        if brightness_std > 0:
            factor = 1.0 + np.random.randn() * brightness_std
            imgs[i] = imgs[i] * factor

    return imgs.reshape(n, -1)


def one_hot(labels, num_classes=10):
    n = labels.shape[0]
    out = np.zeros((n, num_classes))
    out[np.arange(n), labels] = 1.0
    return out


def split_dataset(images, labels, paths, train_ratio=0.7, val_ratio=0.15, seed=42):
    rng = np.random.RandomState(seed)
    n = len(labels)
    indices = rng.permutation(n)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    paths_arr = np.array(paths)

    return {
        'train': (images[train_idx], labels[train_idx], paths_arr[train_idx]),
        'val': (images[val_idx], labels[val_idx], paths_arr[val_idx]),
        'test': (images[test_idx], labels[test_idx], paths_arr[test_idx]),
    }


def prepare_data(data_dir, image_size=64, train_ratio=0.7, val_ratio=0.15, seed=42):
    print("Loading images...")
    images, labels, paths = load_dataset(data_dir, image_size)
    print(f"Loaded {len(labels)} images from {len(CLASS_NAMES)} classes.")

    splits = split_dataset(images, labels, paths, train_ratio, val_ratio, seed)

    train_imgs, train_labels, train_paths = splits['train']
    val_imgs, val_labels, val_paths = splits['val']
    test_imgs, test_labels, test_paths = splits['test']

    train_imgs, mean, std = normalize(train_imgs)
    val_imgs, _, _ = normalize(val_imgs, mean, std)
    test_imgs, _, _ = normalize(test_imgs, mean, std)

    train_X = flatten(train_imgs)
    val_X = flatten(val_imgs)
    test_X = flatten(test_imgs)

    print(f"Train: {train_X.shape[0]}, Val: {val_X.shape[0]}, Test: {test_X.shape[0]}")

    return {
        'train_X': train_X, 'train_y': train_labels, 'train_paths': train_paths,
        'val_X': val_X, 'val_y': val_labels, 'val_paths': val_paths,
        'test_X': test_X, 'test_y': test_labels, 'test_paths': test_paths,
        'mean': mean, 'std': std,
    }