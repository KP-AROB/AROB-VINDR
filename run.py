import argparse
import logging
import os
from src.dataset import VindrDicomDataset
from torch.utils.data import DataLoader
from src.prepare import save_npy_chunk_dataset

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

    train_folder = os.path.join(args.out_dir, 'train')
    test_folder = os.path.join(args.out_dir, 'test')

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # PREPARATION

    train_dataset = VindrDicomDataset(
        args.data_dir, image_size=256, train=True)
    test_dataset = VindrDicomDataset(
        args.data_dir, image_size=256, train=False)

    train_dataloader = DataLoader(
        train_dataset, batch_size=8, shuffle=False, pin_memory=True, num_workers=4)

    test_dataloader = DataLoader(
        test_dataset, batch_size=8, shuffle=False, pin_memory=True, num_workers=4)

    save_npy_chunk_dataset(
        train_dataloader, train_folder, 'train', n_chunks=10)
    save_npy_chunk_dataset(test_dataloader, test_folder, 'test', n_chunks=10)
