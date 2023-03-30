import os
import json
import nibabel as nib
import cv2
import numpy as np

class NiiAnnotator:
    def __init__(self, nii_path):
        self.nii_path = nii_path
        self.nii_data = None
        self.nii_header = None
        self.annotations = {
            'filename': os.path.basename(self.nii_path),
            'regions': []
        }
        
    def load_data(self):
        img = nib.load(self.nii_path)
        self.nii_data = img.get_fdata()
        self.nii_header = img.header
        
    
    def annotate(self):
        for i in range(self.nii_data.shape[2]):
            # Get the 2D slice from the 3D data
            slice_data = self.nii_data[:, :, i]

            # Normalize the slice data to 0-255 range
            slice_data = slice_data / np.max(slice_data) * 255
            slice_data = slice_data.astype(np.uint8)

            # Create a copy of the slice data for annotation
            slice_annotated = cv2.cvtColor(slice_data, cv2.COLOR_GRAY2RGB)

            # Draw existing annotations on the image
            for region in self.annotations['regions'][i]['regions']:
                bbox = region['shape_attributes']
                cv2.rectangle(slice_annotated, (bbox['x'], bbox['y']), (bbox['x'] + bbox['width'], bbox['y'] + bbox['height']), (0, 255, 0), 2)

            # Initialize the list of regions for this slice
            regions = []

            # Create an OpenCV window for annotation
            window_name = 'Annotation - Slice ' + str(i)
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 800, 800)

            # Define the callback function for mouse events
            drawing = False
            ix, iy = -1, -1
            bbox = None
            mask = None
            def draw_annotation(event, x, y, flags, param):
                nonlocal drawing, ix, iy, bbox, mask
                
                if event == cv2.EVENT_LBUTTONDOWN:
                    drawing = True
                    ix, iy = x, y

                elif event == cv2.EVENT_MOUSEMOVE:
                    if drawing:
                        cv2.rectangle(slice_annotated, (ix, iy), (x, y), (0, 255, 0), 2)

                elif event == cv2.EVENT_LBUTTONUP:
                    drawing = False
                    cv2.rectangle(slice_annotated, (ix, iy), (x, y), (0, 255, 0), 2)
                    bbox = [ix, iy, x, y]
                    mask = np.zeros_like(slice_data)
                    mask[iy:y, ix:x] = 1

            # Show the image for annotation and wait for user input
            cv2.imshow(window_name, slice_annotated)
            cv2.setMouseCallback(window_name, draw_annotation)
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s') and bbox is not None:
                    # Save the annotation to the list of regions for this slice
                    region = {
                        'region_attributes': {},
                        'shape_attributes': {
                            'name': 'rect',
                            'x': bbox[0],
                            'y': bbox[1],
                            'width': bbox[2] - bbox[0],
                            'height': bbox[3] - bbox[1]
                        }
                    }
                    regions.append(region)

                    # Save the segmentation mask
                    mask_nii = nib.Nifti1Image(mask, self.nii_header)
                    mask_path = os.path.splitext(self.nii_path)[0] + '_mask_' + str(i) + '.nii'
                    nib.save(mask_nii, mask_path)

                    # Move to the next slice
                    cv2.destroyAllWindows()
                    break


       
    def save_annotations(self, output_path):
        with open(output_path, 'w') as f:
            json.dump(self.annotations, f)


path = 'C:/Users/erios/Downloads/Brain MRI Dataset of Multiple Sclerosis with Consensus Manual Lesion Segmentation and Patient Meta Information'
patient = 'Patient-1/1-T2.nii'

nii_path = os.path.join(path, patient)

annotator = NiiAnnotator(nii_path)
annotator.load_data()

annotator.annotate()

annotator.save_annotations('annotations.json')