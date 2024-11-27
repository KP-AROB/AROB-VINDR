import logging
import ast
import numpy as np
import os
import cv2
from src.utils.image import load_dicom_image
from tqdm import tqdm


def format_category_list(category_list):
    return [category.lower().replace(' ', '_') for category in category_list]


def contains_all_classes(category_list, class_list):
    return any(cls in category_list for cls in class_list)


def replace_categories(df, column, target_categories):
    def replace_if_present(categories):
        for target in target_categories:
            if target in categories:
                return target
        return categories

    df[column] = df[column].apply(
        lambda x: replace_if_present(x) if isinstance(x, list) else x)
    return df


def prepare_vindr_dataframe(df_find, class_list):
    original_df_find_len = len(df_find)

    df_find = df_find.copy()

    df_find['finding_categories'] = df_find['finding_categories'].apply(
        ast.literal_eval)
    df_find['finding_categories'] = df_find['finding_categories'].apply(
        format_category_list)

    # Filter the DataFrame and use .loc to modify values safely
    df_find = df_find[df_find['finding_categories'].apply(
        lambda x: contains_all_classes(x, class_list))]
    logging.info('{} lines were removed from the dataset. The dataset now contains {} images'.format(
        original_df_find_len - len(df_find), len(df_find)))

    df_find = replace_categories(df_find, 'finding_categories', class_list)
    label_to_class_mapping = {class_name: np.int8(
        idx) for idx, class_name in enumerate(df_find['finding_categories'].unique())}

    # Use .loc here to avoid setting on a copy
    df_find.loc[:, 'finding_categories'] = df_find['finding_categories'].map(
        label_to_class_mapping)

    return df_find, label_to_class_mapping


def process_row(data_dir, row):
    """
    Function to load and process a DICOM image.
    Converts the DICOM image into a numpy array of type float32.
    """
    sample_path = os.path.join(
        data_dir, 'images', row['study_id'], row['image_id'] + '.dicom')
    sample_img = load_dicom_image(sample_path)
    sample_img = cv2.resize(
        sample_img,
        (512, 512),
        interpolation=cv2.INTER_LINEAR,
    )
    return sample_img


def save_df_to_npy(df, data_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    images = []
    with tqdm(total=len(df), desc='Preparing dataset') as pbar:
        for _, row in df.iterrows():
            img = process_row(data_dir, row)
            images.append(img)
            pbar.update()

    logging.info(f"Preparing file to save to ...")
    images_np = np.array(images)
    file_path = os.path.join(out_dir, 'vindr-mammo.npy')
    np.save(file_path, images_np)
    logging.info(f"Saved {len(df)} images to {file_path}")
