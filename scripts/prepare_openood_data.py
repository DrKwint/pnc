#!/usr/bin/env python3
"""Download and prepare OpenOOD CIFAR-10 OOD benchmark datasets as .npz files.

Creates the expected directory layout:
  openood_data/cifar10/near_ood/cifar100.npz
  openood_data/cifar10/near_ood/tiny_imagenet.npz
  openood_data/cifar10/far_ood/mnist.npz
  openood_data/cifar10/far_ood/svhn.npz
  openood_data/cifar10/far_ood/textures.npz
  openood_data/cifar10/far_ood/places365.npz

Each .npz has keys: images (N, 32, 32, 3) uint8, labels (N,) int32.
"""

import os
import numpy as np
from pathlib import Path


def _resize_batch(images: np.ndarray, target_size: int = 32) -> np.ndarray:
    """Resize a batch of images to (target_size, target_size) using PIL."""
    from PIL import Image
    out = []
    for img in images:
        pil_img = Image.fromarray(img)
        if pil_img.size != (target_size, target_size):
            pil_img = pil_img.resize((target_size, target_size), Image.BILINEAR)
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        out.append(np.asarray(pil_img))
    return np.stack(out)


def _save_npz(path: str, images: np.ndarray, labels: np.ndarray):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, images=images.astype(np.uint8), labels=labels.astype(np.int32))
    print(f"  Saved {path}: images {images.shape}, labels {labels.shape}")


def prepare_cifar100(root: str):
    """CIFAR-100 test set as near-OOD for CIFAR-10."""
    out_path = os.path.join(root, "cifar10", "near_ood", "cifar100.npz")
    if os.path.exists(out_path):
        print(f"  Already exists: {out_path}")
        return

    import tensorflow as tf
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    (_, _), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
    x_test = x_test.astype(np.uint8)
    y_test = y_test.flatten().astype(np.int32)
    _save_npz(out_path, x_test, y_test)


def prepare_mnist(root: str):
    """MNIST test set as far-OOD for CIFAR-10."""
    out_path = os.path.join(root, "cifar10", "far_ood", "mnist.npz")
    if os.path.exists(out_path):
        print(f"  Already exists: {out_path}")
        return

    import tensorflow as tf
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # Grayscale 28x28 -> resize to 32x32 -> convert to RGB
    from PIL import Image
    images = []
    for img in x_test:
        pil = Image.fromarray(img, mode="L").resize((32, 32), Image.BILINEAR).convert("RGB")
        images.append(np.asarray(pil))
    images = np.stack(images)
    _save_npz(out_path, images, y_test.flatten().astype(np.int32))


def prepare_svhn(root: str):
    """SVHN test set as far-OOD for CIFAR-10."""
    out_path = os.path.join(root, "cifar10", "far_ood", "svhn.npz")
    if os.path.exists(out_path):
        print(f"  Already exists: {out_path}")
        return

    # Download SVHN test set from the standard URL
    import urllib.request
    import scipy.io
    url = "http://ufldl.stanford.edu/housenumbers/test_32x32.mat"
    mat_path = os.path.join(root, "_cache", "svhn_test_32x32.mat")
    Path(mat_path).parent.mkdir(parents=True, exist_ok=True)
    if not os.path.exists(mat_path):
        print(f"  Downloading SVHN test set...")
        urllib.request.urlretrieve(url, mat_path)
    data = scipy.io.loadmat(mat_path)
    # SVHN .mat has X shape (32, 32, 3, N) and y shape (N, 1) with labels 1-10 (10=0)
    x = data["X"].transpose(3, 0, 1, 2).astype(np.uint8)  # (N, 32, 32, 3)
    y = data["y"].flatten().astype(np.int32)
    y[y == 10] = 0
    _save_npz(out_path, x, y)


def prepare_textures(root: str):
    """DTD Textures as far-OOD for CIFAR-10.

    Downloads the DTD dataset and takes all images, resized to 32x32.
    """
    out_path = os.path.join(root, "cifar10", "far_ood", "textures.npz")
    if os.path.exists(out_path):
        print(f"  Already exists: {out_path}")
        return

    import tarfile
    import urllib.request
    from PIL import Image

    url = "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz"
    tar_path = os.path.join(root, "_cache", "dtd.tar.gz")
    extract_dir = os.path.join(root, "_cache", "dtd")
    Path(tar_path).parent.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(tar_path):
        print(f"  Downloading DTD Textures...")
        urllib.request.urlretrieve(url, tar_path)

    if not os.path.exists(extract_dir):
        print(f"  Extracting DTD...")
        with tarfile.open(tar_path) as tf:
            tf.extractall(os.path.join(root, "_cache"))

    # Collect all images from DTD
    images = []
    labels = []
    dtd_images_dir = os.path.join(extract_dir, "images")
    if not os.path.exists(dtd_images_dir):
        dtd_images_dir = os.path.join(root, "_cache", "dtd", "dtd", "images")

    class_idx = 0
    for class_dir in sorted(Path(dtd_images_dir).iterdir()):
        if not class_dir.is_dir():
            continue
        for img_path in sorted(class_dir.glob("*")):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            try:
                with Image.open(img_path) as img:
                    img = img.convert("RGB").resize((32, 32), Image.BILINEAR)
                    images.append(np.asarray(img))
                    labels.append(class_idx)
            except Exception:
                continue
        class_idx += 1

    images = np.stack(images).astype(np.uint8)
    labels = np.array(labels, dtype=np.int32)
    _save_npz(out_path, images, labels)


def prepare_tiny_imagenet(root: str):
    """Tiny ImageNet validation set as near-OOD for CIFAR-10."""
    out_path = os.path.join(root, "cifar10", "near_ood", "tiny_imagenet.npz")
    if os.path.exists(out_path):
        print(f"  Already exists: {out_path}")
        return

    import zipfile
    import urllib.request
    from PIL import Image

    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_path = os.path.join(root, "_cache", "tiny-imagenet-200.zip")
    extract_dir = os.path.join(root, "_cache", "tiny-imagenet-200")
    Path(zip_path).parent.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(zip_path):
        print(f"  Downloading Tiny ImageNet...")
        urllib.request.urlretrieve(url, zip_path)

    if not os.path.exists(extract_dir):
        print(f"  Extracting Tiny ImageNet...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(os.path.join(root, "_cache"))

    # Use validation set (10K images)
    val_dir = os.path.join(extract_dir, "val")
    # Parse val_annotations.txt for labels
    annotations = {}
    ann_path = os.path.join(val_dir, "val_annotations.txt")
    if os.path.exists(ann_path):
        with open(ann_path) as f:
            for line in f:
                parts = line.strip().split("\t")
                annotations[parts[0]] = parts[1]

    images = []
    labels = []
    # Get class to index mapping
    wnids = sorted(set(annotations.values())) if annotations else []
    wnid_to_idx = {w: i for i, w in enumerate(wnids)}

    img_dir = os.path.join(val_dir, "images")
    if os.path.isdir(img_dir):
        for img_name in sorted(os.listdir(img_dir)):
            img_path = os.path.join(img_dir, img_name)
            try:
                with Image.open(img_path) as img:
                    img = img.convert("RGB").resize((32, 32), Image.BILINEAR)
                    images.append(np.asarray(img))
                label = wnid_to_idx.get(annotations.get(img_name, ""), -1)
                labels.append(label)
            except Exception:
                continue

    images = np.stack(images).astype(np.uint8)
    labels = np.array(labels, dtype=np.int32)
    _save_npz(out_path, images, labels)


def prepare_places365(root: str):
    """Places365 test images as far-OOD for CIFAR-10.

    Uses a random 10K subset of the validation set, resized to 32x32.
    """
    out_path = os.path.join(root, "cifar10", "far_ood", "places365.npz")
    if os.path.exists(out_path):
        print(f"  Already exists: {out_path}")
        return

    import tarfile
    import urllib.request
    from PIL import Image

    # Use the small 256x256 validation set
    url = "http://data.csail.mit.edu/places/places365/val_256.tar"
    tar_path = os.path.join(root, "_cache", "places365_val_256.tar")
    extract_dir = os.path.join(root, "_cache", "places365_val")
    Path(tar_path).parent.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(tar_path):
        print(f"  Downloading Places365 validation set (may take a while)...")
        urllib.request.urlretrieve(url, tar_path)

    if not os.path.exists(extract_dir):
        print(f"  Extracting Places365...")
        Path(extract_dir).mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_path) as tf:
            tf.extractall(extract_dir)

    # Collect images and take first 10K
    images = []
    img_root = os.path.join(extract_dir, "val_256")
    if not os.path.isdir(img_root):
        img_root = extract_dir

    all_imgs = sorted(Path(img_root).rglob("*.jpg"))[:10000]
    for img_path in all_imgs:
        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB").resize((32, 32), Image.BILINEAR)
                images.append(np.asarray(img))
        except Exception:
            continue

    images = np.stack(images).astype(np.uint8)
    labels = np.full(len(images), -1, dtype=np.int32)
    _save_npz(out_path, images, labels)


def prepare_cifar10_as_ood(root: str):
    """Save CIFAR-10 test set as near-OOD for CIFAR-100 ID."""
    out_path = os.path.join(root, "cifar100", "near_ood", "cifar10.npz")
    if os.path.exists(out_path):
        print(f"  Already exists: {out_path}")
        return

    import tensorflow as tf
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_test = x_test.astype(np.uint8)
    y_test = y_test.flatten().astype(np.int32)
    _save_npz(out_path, x_test, y_test)


def prepare_cifar100_benchmark(root: str):
    """Set up the CIFAR-100 OOD benchmark layout by reusing CIFAR-10 npz files.

    CIFAR-100's near-OOD set is {cifar10, tiny_imagenet}; far-OOD is {mnist, svhn,
    textures, places365}. The latter five files are identical to those for the
    CIFAR-10 benchmark — just hard-linked into the cifar100 directory tree.
    """
    # near-OOD: cifar10 (new), tiny_imagenet (link from cifar10/near_ood)
    prepare_cifar10_as_ood(root)

    src_dst_pairs = [
        ("cifar10/near_ood/tiny_imagenet.npz", "cifar100/near_ood/tiny_imagenet.npz"),
        ("cifar10/far_ood/mnist.npz", "cifar100/far_ood/mnist.npz"),
        ("cifar10/far_ood/svhn.npz", "cifar100/far_ood/svhn.npz"),
        ("cifar10/far_ood/textures.npz", "cifar100/far_ood/textures.npz"),
        ("cifar10/far_ood/places365.npz", "cifar100/far_ood/places365.npz"),
    ]
    for src_rel, dst_rel in src_dst_pairs:
        src = os.path.join(root, src_rel)
        dst = os.path.join(root, dst_rel)
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        if os.path.exists(dst):
            print(f"  Already linked: {dst}")
            continue
        if not os.path.exists(src):
            raise FileNotFoundError(f"Source missing: {src}. Run main() first.")
        # Use a hard link rather than symlink so the loader can stat() reliably
        os.link(src, dst)
        print(f"  Linked {src} -> {dst}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cifar100",
        action="store_true",
        help="Also prepare CIFAR-100 OOD benchmark layout.",
    )
    parser.add_argument(
        "--only-cifar100",
        action="store_true",
        help="Skip CIFAR-10 prep and only set up CIFAR-100 layout.",
    )
    args = parser.parse_args()

    root = "openood_data"

    if not args.only_cifar100:
        print("Preparing OpenOOD CIFAR-10 benchmark datasets...")

        print("\n[1/6] CIFAR-100 (near-OOD)")
        prepare_cifar100(root)

        print("\n[2/6] Tiny ImageNet (near-OOD)")
        prepare_tiny_imagenet(root)

        print("\n[3/6] MNIST (far-OOD)")
        prepare_mnist(root)

        print("\n[4/6] SVHN (far-OOD)")
        prepare_svhn(root)

        print("\n[5/6] Textures / DTD (far-OOD)")
        prepare_textures(root)

        print("\n[6/6] Places365 (far-OOD)")
        prepare_places365(root)

        print("\nCIFAR-10 prep done.")

    if args.cifar100 or args.only_cifar100:
        print("\nPreparing CIFAR-100 OOD benchmark layout...")
        prepare_cifar100_benchmark(root)
        print("CIFAR-100 prep done.")

    print("\nDone! All datasets saved under", root)


if __name__ == "__main__":
    main()
