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

To use prepare the dataset as .npy files you can run the following :

```bash
python run.py --data_dir path/to/vindr --out_dir path/to/npy
```
