import os
import pandas as pd
from shutil import copyfile
from torchvision.datasets import ImageFolder

class MRIImageFolder:
    def __init__(self, csv_path, mri_dir, output_dir):
        self.csv_path = csv_path
        self.mri_dir = mri_dir
        self.output_dir = output_dir
        self.mri_imagefolder = None
        self.mask_imagefolder = None

    def create_directories(self):
        for class_name in ["Class 0", "Class 1", "Class 2"]:
            os.makedirs(os.path.join(self.output_dir, "MRI", class_name), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, "Mask", class_name), exist_ok=True)

    def convert_to_imagefolder(self):
        df = pd.read_csv(self.csv_path)
        total_patients = len(df)
        processed_patients = 0
        for _, row in df.iterrows():
            patient_id = row["ID"]
            class_label = row["EDSS_cat"]
            patient_dir = os.path.join(self.mri_dir, f"Patient-{patient_id}")
            num_files = 0
            for sequence in ["T1", "T2", "Flair"]:
                input_path = os.path.join(patient_dir, f"{patient_id}-{sequence}.nii")
                output_path = os.path.join(self.output_dir, "MRI", f"Class {class_label}", f"{patient_id}-{sequence}.nii")
                copyfile(input_path, output_path)
                num_files += 1
            for sequence in ["T1", "T2", "Flair"]:
                input_path = os.path.join(patient_dir, f"{patient_id}-LesionSeg-{sequence}.nii")
                output_path = os.path.join(self.output_dir, "Mask", f"Class {class_label}", f"{patient_id}-LesionSeg-{sequence}.nii")
                copyfile(input_path, output_path)
                num_files += 1
            processed_patients += 1
            print(f"Processed patient {patient_id} ({num_files} files). {processed_patients} of {total_patients} patients.")
        # Create ImageFolders for the MRI scans and lesion segmentation masks
        # self.mri_imagefolder = ImageFolder(os.path.join(self.output_dir, "MRI"))
        # self.mask_imagefolder = ImageFolder(os.path.join(self.output_dir, "Mask"))
