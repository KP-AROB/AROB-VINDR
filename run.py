import argparse
import logging
import numpy as np
import os
from src.dataset import VindrDicomDataset
from tqdm import tqdm
from torch.utils.data import DataLoader


def save_dataset(dataloader, mode='train'):
    images_list = []
    labels_list = []

    with tqdm(total=len(dataloader), desc=f"Preparing {mode} images and labels") as pbar:
        for images, labels in dataloader:
            images_list.append(images.numpy())
            labels_list.append(labels)
            pbar.update(1)

    images_np = np.concatenate(images_list, axis=0)
    labels_np = np.concatenate(labels_list, axis=0)

    np.save(os.path.join(args.out_dir, f'vindr_images_{mode}.npy'), images_np)
    np.save(os.path.join(args.out_dir, f'vindr_labels_{mode}.npy'), labels_np)
    logging.info(
        f"Saved images to {os.path.join(args.out_dir, f'vindr_images_{mode}.npy')}")
    logging.info(
        f"Saved labels to {os.path.join(args.out_dir, f'vindr_labels_{mode}.npy')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Vindr Mammo data preparation")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default='./data')
    args = parser.parse_args()
    parser.set_defaults(synthetize=False)

    # INIT
    logging_message = "[AROB-2025-KAPTIOS-VINDR]"

    logging.basicConfig(
        level=logging.INFO,
        format=f'%(asctime)s - {logging_message} - %(levelname)s - %(message)s'
    )
    logging.info('Running Vindr Mammo dataset preparation')

    os.makedirs(args.out_dir, exist_ok=True)

    # PREPARATION

    train_dataset = VindrDicomDataset(
        args.data_dir, image_size=256, train=True)
    test_dataset = VindrDicomDataset(
        args.data_dir, image_size=256, train=False)

    train_dataloader = DataLoader(
        train_dataset, batch_size=8, shuffle=False, pin_memory=True, num_workers=4)

    test_dataloader = DataLoader(
        train_dataset, batch_size=8, shuffle=False, pin_memory=True, num_workers=4)

    save_dataset(train_dataloader, 'train')
    save_dataset(test_dataloader, 'test')
