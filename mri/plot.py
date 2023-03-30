import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import math

class MRIViewer:
    def __init__(self, input_dir):
        self.input_dir = input_dir
        self.patient = None
        self.img = None
        self.data = None
        self.header = None
        self.fig = None
        self.ax = None
        self.seg_img = None
        
    def load_patient(self, patient_path):
        self.patient = patient_path
        self.img = nib.load(os.path.join(self.input_dir, self.patient))
        self.data = self.img.get_fdata()
        self.header = self.img.header
        
    def plot_patient(self, n_cols=5):
        # Calculate the number of rows and columns needed
        n_images = self.data.shape[2]
        n_rows = math.ceil(n_images / n_cols)

        # Create a figure with the right number of rows and columns
        self.fig, self.ax = plt.subplots(n_rows, n_cols, figsize=(n_cols*3, n_rows*3))

        # Flatten the axis array for ease of indexing
        self.ax = self.ax.flatten()

        # Set the title of the plot to the name of the .nii file and remove the file extension
        self.fig.suptitle(self.patient.split('/')[-1].split('.')[0])

        # Loop over each image and plot it in the appropriate subplot
        for i in range(n_images):
            # Plot the original image
            self.ax[i].imshow(self.data[:,:,i], cmap='gray')
            self.ax[i].axis('off')  # Hide the axes labels

            # Check if there is a corresponding segmentation image
            if self.seg_img is not None and self.seg_img.shape[2] == n_images:
                # Get the segmentation mask for the current slice
                seg_mask = self.seg_img[:, :, i]

                # Overlay the segmentation mask on top of the original image using alpha blending
                self.ax[i].imshow(np.ma.masked_where(seg_mask == 0, seg_mask), cmap="RdBu", alpha=1)

        # Remove any unused subplots
        for i in range(n_images, n_rows*n_cols):
            self.fig.delaxes(self.ax[i])

        # Adjust the layout of the plot to ensure the title is visible
        plt.tight_layout()

        # Show the plot
        plt.show()

    def load_segmentation(self, seg_path):
        self.seg_img = nib.load(os.path.join(self.input_dir, seg_path)).get_fdata()
        
    def save_plot(self, output_path):
        self.fig.savefig(output_path)


# viewer = MRIViewer('Brain MRI Dataset of Multiple Sclerosis with Consensus Manual Lesion Segmentation and Patient Meta Information')
# viewer.load_patient('Patient-1/1-T2.nii')
# viewer.load_segmentation('Patient-1/1-LesionSeg-T2.nii')
# viewer.plot_patient(n_cols=5)
