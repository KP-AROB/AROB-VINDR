import os
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import ast
from pydicom import dcmread
from pydicom.pixel_data_handlers import apply_voi_lut
import numpy as np


class VindrDicomDataset(Dataset):
    def __init__(self,
                 data_dir,
                 class_list=['mass', 'suspicious_calcification', 'no_finding'],
                 resize=512):

        self.df = self.__prepare_vindr_dataframe(data_dir, class_list)
        self.class_list = class_list
        self.data_dir = data_dir
        self.resize = resize
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((resize, resize))
        ])

    def __load_dicom_image(self, path):
        ds = dcmread(path)
        img2d = ds.pixel_array
        img2d = apply_voi_lut(img2d, ds)

        if ds.PhotometricInterpretation == "MONOCHROME1":
            img2d = np.amax(img2d) - img2d

        img2d = img2d.astype(np.float32)
        normalized_data = (img2d - np.min(img2d)) / \
            (np.max(img2d) - np.min(img2d))
        return normalized_data

    def __prepare_vindr_dataframe(self, data_dir, class_list):

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

        df_find = pd.read_csv(os.path.join(
            data_dir, 'finding_annotations.csv'))
        df_find['finding_categories'] = df_find['finding_categories'].apply(
            ast.literal_eval)
        df_find['finding_categories'] = df_find['finding_categories'].apply(
            format_category_list)
        df_find = df_find[df_find['finding_categories'].apply(
            lambda x: contains_all_classes(x, class_list))]
        df_find = replace_categories(df_find, 'finding_categories', class_list)
        return df_find

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_path = os.path.join(
            self.data_dir, 'images', row['study_id'], row['image_id'] + '.dicom')

        img = self.__load_dicom_image(sample_path)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img_tensor = self.transform(img)
        label = self.class_list.index(row['finding_categories'])
        return img_tensor, label
