from tqdm import tqdm
import numpy as np
import os


def save_npy_chunk_dataset(dataloader, out_dir: str, mode='train', n_chunks=2):
    """
    Saves images and labels from dataloader in chunks.

    Args:
        dataloader: PyTorch dataloader
        mode: 'train' or 'test' (default: 'train')
        chunk_size: Number of images to save in each chunk (default: 1000)
    """
    total_batches = len(dataloader)
    batches_per_chunk = total_batches // n_chunks
    remainder = total_batches % n_chunks

    chunk_sizes = [batches_per_chunk + 1 if i <
                   remainder else batches_per_chunk for i in range(n_chunks)]

    chunk_images = []
    chunk_labels = []
    current_chunk = 0
    batches_in_current_chunk = 0

    with tqdm(total=total_batches, desc=f"Saving {mode} data chunk {current_chunk + 1} / {n_chunks}") as pbar:
        for _, (images, labels) in enumerate(dataloader):
            chunk_images.append(images.numpy())
            chunk_labels.append(labels)
            batches_in_current_chunk += 1
            # Save chunk when enough batches are collected
            if batches_in_current_chunk == chunk_sizes[current_chunk]:
                np.save(os.path.join(out_dir, f"{mode}_images_chunk_{current_chunk}.npy"), np.concatenate(
                    chunk_images, axis=0))
                np.save(os.path.join(out_dir, f"{mode}_labels_chunk_{current_chunk}.npy"), np.concatenate(
                    chunk_labels, axis=0))
                pbar.set_description(
                    f"Saving data chunk {current_chunk + 1} / {n_chunks}")
                chunk_images = []
                chunk_labels = []
                batches_in_current_chunk = 0
                current_chunk += 1
            pbar.update(1)
