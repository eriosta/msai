import os
import pandas as pd
from shutil import copyfile

class MRIImageFolder:
    def __init__(self, csv_path, mri_dir, output_dir):
        self.csv_path = csv_path
        self.mri_dir = mri_dir
        self.output_dir = output_dir

    def create_directories(self):
        for class_name in ["Class 0", "Class 1", "Class 2"]:
            os.makedirs(os.path.join(self.output_dir, class_name), exist_ok=True)

    def convert_to_imagefolder(self):
        df = pd.read_csv(self.csv_path)
        for _, row in df.iterrows():
            patient_id = row["ID"]
            class_label = row["EDSS_cat"]
            patient_dir = os.path.join(self.mri_dir, f"Patient-{patient_id}")
            for sequence in ["T1", "T2", "Flair"]:
                input_path = os.path.join(patient_dir, f"{patient_id}-{sequence}.nii")
                output_path = os.path.join(self.output_dir, f"Class {class_label}", f"{patient_id}-{sequence}.nii")
                copyfile(input_path, output_path)