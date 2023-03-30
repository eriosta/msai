import os
import numpy as np
import nibabel as nib
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.applications import vit

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
        nii_files = os.listdir(self.nii_dir)
        data = []
        masks = []
        for nii_file in nii_files:
            if nii_file.endswith('.nii'):
                nii_path = os.path.join(self.nii_dir, nii_file)
                nii_data = nib.load(nii_path).get_fdata()
                slices = np.rollaxis(nii_data, 2)  # Roll the axis to get a list of slices
                data.extend(slices)

                # Load the corresponding segmentation mask for each slice
                mask_path = os.path.splitext(nii_path)[0] + '_mask.nii'
                mask_data = nib.load(mask_path).get_fdata()
                mask_slices = np.rollaxis(mask_data, 2)
                masks.extend(mask_slices)
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

    def load_model(self):
        # Load ViT model and pre-trained weights
        self.model = vit.ViT(image_size=self.img_size, classes=self.classes, include_top=self.include_top, pretrained=self.pretrained)

    def add_attention_mask(self):
        attention_mask = tf.keras.layers.Input(shape=(self.img_size, self.img_size, 1))
        x = tf.keras.layers.Concatenate()([self.model.output, attention_mask])
        self.model = tf.keras.Model(inputs=[self.model.input, attention_mask], outputs=x)

    def train_model(self, epochs=10):
        # Fine-tune ViT model on training set
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.history = self.model.fit([self.X_train, self.y_train], self.y_train, epochs=epochs, validation_data=([self.X_val, self.y_val], self.y_val))
        
        # Save model and weights
        model_dir = os.path.join(self.nii_dir, 'models')
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f'vit_model_{self.img_size}.h5')
        self.model.save(model_path)
        
        weights_dir = os.path.join(self.nii_dir, 'weights')
        os.makedirs(weights_dir, exist_ok=True)
        weights_path = os.path.join(weights_dir, f'vit_weights_{self.img_size}.h5')
        self.model.save_weights(weights_path)
