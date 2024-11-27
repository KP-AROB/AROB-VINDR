import h5py
from src.utils.image import load_and_process_images
import numpy as np
from tqdm import tqdm
import logging


def create_h5_file_from_dataframe(df, data_dir, output_h5_file, batch_size=10):
    """
    Function to create an HDF5 file containing DICOM images as np.float32 arrays.

    df: DataFrame containing 'file_path' and 'label' columns.
    output_h5_file: Path to the output H5 file.
    """
    num_samples = len(df)
    labels = df['finding_categories'].values

    # Initialize HDF5 file and datasets
    with h5py.File(output_h5_file, 'a') as h5f:
        # Initialize datasets for images and labels
        images_dataset = h5f.create_dataset(
            'images', (num_samples,), dtype=np.float32)
        labels_dataset = h5f.create_dataset(
            'labels', (num_samples,), dtype=np.int32)

        # Load and process images in parallel
        images = load_and_process_images(df, data_dir, batch_size)

        # Write images and labels to the HDF5 file
        with tqdm(total=num_samples, desc="Saving to HDF5") as pbar:
            for i in range(num_samples):
                images_dataset[i] = images[i]
                labels_dataset[i] = labels[i]
                pbar.update()
        logging.info(f"Data saved to {output_h5_file}")
