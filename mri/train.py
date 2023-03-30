import os
import numpy as np
import nibabel as nib
import glob
from sklearn.model_selection import train_test_split
import timm
from timm.models import vision_transformer as vit


class ViTTrainer:
    def __init__(self, nii_dir, img_size=224, classes=1, include_top=False, pretrained=True, epochs=10):
        self.nii_dir = nii_dir
        self.img_size = img_size
        self.classes = classes
        self.include_top = include_top
        self.pretrained = pretrained
        self.epochs = epochs
        self.data = None
        self.masks = None
        self.model = None
        self.history = None

    def load_nii_data(self):
        mri_folder = os.path.join(self.nii_dir, 'MRI')
        mask_folder = os.path.join(self.nii_dir, 'Mask')
        data = []
        masks = []
        mri_paths = []
        mask_paths = []

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

                    # Get the file names without the patient ID
                    mri_file_name = os.path.basename(mri_path)
                    mask_file_name = os.path.basename(mask_path)
                    mri_file_name = mri_file_name.split('-', 1)[1]
                    mask_file_name = mask_file_name.split('-', 1)[1]

                    # Save the file paths without the patient ID
                    mri_paths.append(os.path.join(mri_folder, class_dir, mri_file_name))
                    mask_paths.append(os.path.join(mask_folder, class_dir, mask_file_name))

        self.data = np.array(data)
        self.masks = np.array(masks)

    def preprocess_data(self):
        data_resized = np.array([tf.image.resize(slice, (self.img_size, self.img_size)) for slice in self.data])
        data_rescaled = Rescaling(scale=1./255)(data_resized)  # Normalize pixel values to [0, 1]

        masks_resized = np.array([tf.image.resize(slice, (self.img_size, self.img_size), method='nearest') for slice in self.masks])
        masks_rescaled = masks_resized / np.max(masks_resized)  # Normalize pixel values to [0, 1]

        # Split data into training, validation, and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(data_rescaled, masks_rescaled, test_size=0.2, random_state=42)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, test_size=0.2, random_state=42)

    def load_model(self, model_path=None, weights_path=None):
            # Load ViT model and pre-trained weights
            self.model = timm.create_model('vit_base_patch16_224', num_classes=self.classes, pretrained=self.pretrained)

            # Load saved weights if provided
            if weights_path:
                self.model.load_weights(weights_path)

            # Load saved model if provided
            if model_path:
                self.model = tf.keras.models.load_model(model_path, compile=False)

            # Add attention mask to ViT model
            attention_mask = tf.keras.layers.Input(shape=(self.img_size, self.img_size, 1))
            x = tf.keras.layers.Concatenate()([self.model.output, attention_mask])
            self.model = tf.keras.Model(inputs=[self.model.input, attention_mask], outputs=x)
