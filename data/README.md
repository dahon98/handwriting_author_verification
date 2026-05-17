# Handwriting Verification Dataset Pairs

This repository contains precomputed pair files for training, validation, and testing of handwriting verification models (ResNet-50, CNN-CrossViT, and other architectures).

## File Descriptions

| File Name | Description |
|:---|:---|
| `cvl_pairs.pkl` | Precomputed 5000 image pairs for the CVL (Computer Vision Lab) dataset, containing continuous handwritten text in German and English. Used for out-of-domain evaluation. |
| `icdar_chinese_pairs.pkl` | Precomputed 5000 image pairs for the ICDAR 2011 SigComp Chinese subset. Contains genuine and skilled forgery signatures. Used for out-of-domain evaluation on non-Latin script. |
| `icdar_dutch_pairs.pkl` | Precomputed 5000 image pairs for the ICDAR 2011 SigComp Dutch subset. Contains genuine and skilled forgery signatures. Used for out-of-domain evaluation. |
| `test_pairs.pkl` | Test set 1000 pairs from the Training Domain (IAM, CEDAR, Hindi, Bengali). Used for final evaluation of model performance. |
| `test_pairs_vit.pkl` | Test set 1000 pairs specifically formatted for Vision Transformer (ViT) based architectures (e.g., CNN-CrossViT). |
| `val_pairs.pkl` | Validation set 2000 pairs from the Training Domain. Used for hyperparameter tuning, threshold selection, and early stopping. |
| `val_pairs_vit.pkl` | Validation set 2000 pairs specifically formatted for Vision Transformer (ViT) based architectures. |

## File Format

Each `.pkl` file is a serialized Python list (pickle format) containing tuples of:

```python
(image1_path, image2_path, label)
```

## Trainig Domain
Download link: https://drive.google.com/file/d/1A-MZ6zw_0uJQPuwj9GFeqwOPz2fqKOn_/view?usp=drive_link. Due to size cannot be uploaded to github.

The Training Domain contains images from four datasets (IAM, CEDAR, Hindi, Bengali) used for model training and validation. 

## ICDAR (Chinese) Dataset
Download link: https://drive.google.com/file/d/1acZ-ivWmnOKhajHI0Kxi4WA6cA3Nibg1/view?usp=drive_link

Chinese signature subset from ICDAR 2011 SigComp competition. Contains genuine signatures and skilled forgeries.

## ICDAR (Dutch) Dataset
Download link: https://drive.google.com/file/d/1libR_GaMix9vkOgshT-5F_gI-fXd5NdC/view?usp=drive_link

Dutch signature subset from ICDAR 2011 SigComp competition. Contains genuine signatures and skilled forgeries.

## CVL Dataset
Download link: https://drive.google.com/file/d/12ef4yi_hkA3EBOVie3xoEOyRWktPBK7A/view?usp=drive_link

Computer Vision Lab dataset with continuous handwritten text in German and English.

## Error Analysis Data
Error on test set
Download link: https://drive.google.com/file/d/1xtZcmWtENt3a4099ymdm9VxU6MnVMLDL/view?usp=drive_link

Error on ICDAR-Chinese
Download link: https://drive.google.com/file/d/1DqJLS9ssybg6ZZWtVlh5ygLNOuNhc5zA/view?usp=drive_link

Error on ICDAR-Dutch
Download link: https://drive.google.com/file/d/1g1aV1VZbzmnw6f9me9_wwHJTFlMXx07q/view?usp=drive_link

Error on CVL
Download link: https://drive.google.com/file/d/16uqOxAe0egH-WAG5gPGK3jx_563JiR8Q/view?usp=drive_link

These files are useful for analyzing model weaknesses and improving performance.
