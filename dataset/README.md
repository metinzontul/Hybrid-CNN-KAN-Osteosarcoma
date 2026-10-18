## Dataset
The experiments were conducted using the publicly available Bone Tumor X-ray Radiograph Dataset (BTXRD). Due to GitHub's storage limits, the dataset is not hosted in this repository.

1. Download the original dataset from [Figshare](https://doi.org/10.6084/m9.figshare.27865398).
2. Extract and organize the dataset strictly into the following directory structure before running `train.py`:

dataset/
├── train/
│   ├── osteosarcoma/
│   └── non-osteosarcoma/
├── val/
│   ├── osteosarcoma/
│   └── non-osteosarcoma/
└── test/
    ├── osteosarcoma/
    └── non-osteosarcoma/