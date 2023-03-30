import os
import numpy as np
import nibabel as nib
import glob
from sklearn.model_selection import train_test_split
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from timm.models import vision_transformer as vit
import torch
from PIL import Image
from torchvision.transforms import Resize, Normalize


class BrainDataset(Dataset):
    def __init__(self, data, masks, transform=None):
        self.data = data
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data_slice = self.data[idx]
        mask_slice = self.masks[idx]

        if self.transform:
            data_slice = self.transform(data_slice)
            mask_slice = self.transform(mask_slice)

        return data_slice, mask_slice


class ViTTrainer:
    def __init__(self, nii_dir, img_size=224, classes=1, include_top=False, pretrained=True, epochs=10, batch_size=8):
        self.nii_dir = nii_dir
        self.img_size = img_size
        self.classes = classes
        self.include_top = include_top
        self.pretrained = pretrained
        self.epochs = epochs
        self.batch_size = batch_size
        self.data = None
        self.masks = None
        self.model = None
        self.history = None

    def load_nii_data(self):
        mri_folder = os.path.join(self.nii_dir, 'MRI')
        mask_folder = os.path.join(self.nii_dir, 'Mask')
        data = []
        masks = []

        for class_dir in os.listdir(os.path.join(self.nii_dir, "MRI")):
            for sequence in ["T1", "T2", "Flair"]:
                # Find all MRI files for the current class and sequence
                mri_files = sorted(glob.glob(os.path.join(mri_folder, class_dir, f"*-{sequence}.nii")))

                # Find all corresponding mask files
                mask_files = sorted(glob.glob(os.path.join(mask_folder, class_dir, f"*-LesionSeg-{sequence}.nii")))

                # Load MRI data and corresponding masks
                for mri_path, mask_path in zip(mri_files, mask_files):
                    mri_data = nib.load(mri_path).get_fdata()
                    slices = np.rollaxis(mri_data, 2)  # Roll the axis to get a list of slices
                    data.extend(slices)

                    mask_data = nib.load(mask_path).get_fdata()
                    mask_slices = np.rollaxis(mask_data, 2)
                    masks.extend(mask_slices)

        self.data = np.array(data)
        self.masks = np.array(masks)

    def preprocess_data(self):
            data_resized = np.array([Resize(self.img_size)(Image.fromarray((slice*255).astype(np.uint8))) for slice in self.data])
            data_rescaled = Normalize(mean=[0.5], std=[0.5])(torch.Tensor(data_resized)).numpy()

            masks_resized = np.array([Resize(self.img_size)(Image.fromarray(slice)) for slice in self.masks])
            masks_rescaled = (masks_resized / np.max(masks_resized)).astype(np.uint8)

            # Split data into training, validation, and testing sets
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data_rescaled, masks_rescaled, test_size=0.2, random_state=42)
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, test_size=0.2, random_state=42)