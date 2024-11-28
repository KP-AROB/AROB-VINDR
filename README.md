# Vindr Mammo PyTorch Dataset

This repo offers a Torch Dataset class that prepares the Vindr-Mammo dataset as a list of images with associated classes
The code filters the original finding_annotations.csv file given by the authors to only create a Dataset with a given list of classes.

These classes are : 

- mass
- global_asymmetry
- architectural_istortion
- nipple_etraction
- suspicious_calcification
- focal_asymmetry
- asymmetry
- skin_thickening
- suspicious_lymph_node
- skin_retraction
- no_finding

## Usage

To use the dataset class, you only need to instanciate it by giving it the path to the .csv file (```data_dir```) and the list of classes to keep in the Dataset.

```python
from dataset import VindrDicomDataset
import pandas as pd
import os

data_dir = './data'
class_list = ['mass', 'suspicious_calcification', 'no_finding']
dataset = VindrDicomDataset(data_dir, class_list)
```
