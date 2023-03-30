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
        total_patients = len(df)
        processed_patients = 0
        for _, row in df.iterrows():
            patient_id = row["ID"]
            class_label = row["EDSS_cat"]
            patient_dir = os.path.join(self.mri_dir, f"Patient-{patient_id}")
            for filename in os.listdir(patient_dir):
                if "-LesionSeg-" in filename:
                    sequence = filename.split("-LesionSeg-")[-1].replace(".nii", "")
                else:
                    sequence = None
                    for s in ["T1", "T2", "Flair"]:
                        if s in filename:
                            sequence = s
                            break
                if sequence:
                    input_path = os.path.join(patient_dir, filename)
                    output_path = os.path.join(self.output_dir, f"Class {class_label}", f"{patient_id}-{sequence}.nii")
                    copyfile(input_path, output_path)
                    processed_patients += 1
                    if processed_patients % 10 == 0:
                        print(f"Processed {processed_patients} of {total_patients} patients.")
