import json
import os
from typing import Self
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class ImageAnalyzer:
    def __init__(self, calibration_file:str = None, images_folder_path:str = None):
        if calibration_file is None:
            raise ValueError("calibration_file is required")
        with open(calibration_file, 'r') as f:
            calibration_data = json.load(f)
        self.grid_size = calibration_data['grid_size']
        self.roi_side_length = calibration_data['roi_side_length']
        self.calibration_date = calibration_data['calibration_date']
        self.images_folder_path = images_folder_path
        self.atoms = []
        print("images analyzed with data taken on: ", self.calibration_date)
        print("expecting an atom array of size: ", self.grid_size[0], "x", self.grid_size[1])
        print("the ROIs are of size: ", self.roi_side_length, "x", self.roi_side_length)
        print("images to analyze are in: ", self.images_folder_path)

        for i in range(self.grid_size[0]*self.grid_size[1]):
            self.atoms.append(calibration_data[f"atom {i}"])

    def count_atoms_in_image(self, image_path:str):
        atom_count = 0
        image = Image.open(os.path.join(self.images_folder_path, image_path))
        pixel_values = np.array(image)
        pixel_values = pixel_values.astype(np.float32)
        for atom in self.atoms:
            if pixel_values[atom["position_y"]-self.roi_side_length//2:atom["position_y"]-self.roi_side_length//2+self.roi_side_length, atom["position_x"]-self.roi_side_length//2:atom["position_x"]-self.roi_side_length//2+self.roi_side_length].sum() > atom["threshold"]:
                atom_count += 1
        # print(f"atom count in {image_path} is {atom_count}")
        return atom_count
    
    def atoms_count(self,images_per_value:int = 10, plot_atom_counts:bool = False, title:str = None, x_label:str = None, y_label:str = None):
        atom_counts_before = []
        atom_counts_after = []
        variable_counter = 0
        atoms_count_per_value = []

        total_images = len(os.listdir(self.images_folder_path))
        images = sorted(os.listdir(self.images_folder_path))
        for image_index, image_path in enumerate(images):
            if image_index % 20 == variable_counter:
                atom_counts.append(self.count_atoms_in_image(image_path))
            else:continue
        if plot_atom_counts:
            plt.plot(atom_counts)
            if title is not None:
                plt.title(title)
            if x_label is not None:
                plt.xlabel(x_label)
            if y_label is not None:
                plt.ylabel(y_label)
            plt.savefig("atom_counts.png")
        return atom_counts
    
    def atoms_survival_ratio(self, plot_atoms_survival_ratio:bool = False):
        ...
        


def main():
    images_folder_path ="/home/ohmorig-neo/image_analyzer/Jan09_2026_180036first_mw_resonance"
    analyzer = ImageAnalyzer(calibration_file="calibration.json", images_folder_path=images_folder_path)
    analyzer.atoms_count(plot_atom_counts=True, title="Atom Counts", x_label="frequency", y_label="Atom Count")

if __name__ == "__main__":
    main()