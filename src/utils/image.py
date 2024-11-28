from pydicom import dcmread
from pydicom.pixel_data_handlers import apply_voi_lut
import numpy as np


def load_dicom_image(path):
    ds = dcmread(path)
    img2d = ds.pixel_array
    img2d = apply_voi_lut(img2d, ds)

    if ds.PhotometricInterpretation == "MONOCHROME1":
        img2d = np.amax(img2d) - img2d

    img2d = img2d.astype(np.float32)
    normalized_data = (img2d - np.min(img2d)) / (np.max(img2d) - np.min(img2d))
    return normalized_data


def extract_patch(image, xmin, xmax, ymin, ymax, padding=100):
    ymin = max(0, ymin - padding)
    xmin = max(0, xmin - padding)
    ymax = min(image.shape[0], ymax + padding)
    xmax = min(image.shape[1], xmax + padding)
    return image[ymin:ymax, xmin:xmax]
