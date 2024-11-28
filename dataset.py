import os
from torch.utils.data import Dataset
from torchvision import transforms
from src.utils.dataframe import prepare_vindr_dataframe
from src.utils.image import load_dicom_image
import pandas as pd


class VindrDicomDataset(Dataset):
    def __init__(self, data_dir, class_list, resize=512):
        df = pd.read_csv(os.path.join(data_dir, 'finding_annotations.csv'))
        self.df = prepare_vindr_dataframe(df, class_list)
        self.class_list = class_list
        self.data_dir = data_dir
        self.resize = resize
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((resize, resize))
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_path = os.path.join(
            self.data_dir, 'images', row['study_id'], row['image_id'] + '.dicom')

        img = load_dicom_image(sample_path)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img_tensor = self.transform(img)
        label = self.class_list.index(row['finding_categories'])
        return img_tensor, label
