import json
import os
from typing import List
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd

class ImageAnalyzer:
    def __init__(self, 
    calibration_file:str = None, 
    images_folder_path:str = None,
    scan_order:str = None, #"params first" or "reps first"
    n_reps:int = None, #number of repetitions of the same parameter value
    n_parameters:int = None, #number of parameters to scan
    images_per_cycle:int = None, #number of images per cycle
    parameters:list[float] = None, #list of parameter values
    parameter_name:str = None, #name of the parameter
    ):

        if calibration_file is None:
            raise ValueError("calibration_file is required")

        with open(calibration_file, 'r') as f:
            calibration_data = json.load(f)

        self.grid_size = calibration_data['grid_size']
        self.roi_side_length = calibration_data['roi_side_length']
        self.calibration_date = calibration_data['calibration_date']
        self.images_folder_path = images_folder_path
        self.atoms = []
        self.scan_order = scan_order
        self.n_reps = n_reps
        self.n_parameters = n_parameters
        self.images_per_cycle = images_per_cycle
        self.parameters = parameters
        self.parameter_name = parameter_name
        print("images analyzed with data taken on: ", self.calibration_date)
        print("expecting an atom array of size: ", self.grid_size[0], "x", self.grid_size[1])
        print("the ROIs are of size: ", self.roi_side_length, "x", self.roi_side_length)
        print("images to analyze are in: ", self.images_folder_path)

        for i in range(self.grid_size[0]*self.grid_size[1]):
            self.atoms.append(calibration_data[f"atom {i}"])

    def image_info(self, img_idx):
        """ returns the parameter index, the repetition index, and the image type (final or initial) for the given image index"""
        cycle = img_idx // self.images_per_cycle
        image_type = 1 if (img_idx % self.images_per_cycle == self.images_per_cycle - 1) else 0
        
        if self.scan_order == "params first":
            # Sequence: p0_r0, p0_r1, p0_r2, ..., p1_r0, p1_r1, ...
            param_idx = cycle // self.n_reps
            rep_idx = cycle % self.n_reps
        else:  # "rep_first"
            # Sequence: p0_r0, p1_r0, p2_r0, ..., p0_r1, p1_r1, ...
            rep_idx = cycle // self.n_parameters
            param_idx = cycle % self.n_parameters
        
        return param_idx, rep_idx, image_type

    def analyze_images(self):
        images = sorted(os.listdir(self.images_folder_path))
        rows = []
        print(len(images))
        for image_idx, image in enumerate(images):
            param_idx, rep_idx, image_type = self.image_info(image_idx)
            atoms_in_image = self.atoms_in_image(image)
            rows.append({"parameter_index": param_idx, "repetition_index": rep_idx, "image_type": image_type, "atoms_in_image": atoms_in_image})
            
        self.images_data = pd.DataFrame(rows)

    def plot_graph(self, x_data, y_data, x_label, y_label, title, save_path):
        plt.figure()
        plt.plot(x_data, y_data,marker='o')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.savefig(save_path)
        # plt.show()
    
    def plot_survival_vs_parameter(self):
        survival_ratios = []
        for param_idx in range(self.n_parameters):
        #     for rep_idx in range(self.n_reps):

            mask = (self.images_data["parameter_index"] == param_idx)
            ref_occupancy = self.images_data[mask & (self.images_data["image_type"] == 0)]["atoms_in_image"].sum()
            final_occupancy = self.images_data[mask & (self.images_data["image_type"] == 1)]["atoms_in_image"].sum()
            survival_ratio = self.atoms_survival_ratio(ref_occupancy, final_occupancy)
            survival_ratios.append(survival_ratio)

        self.plot_graph(self.parameters, 
        survival_ratios, 
        self.parameter_name, 
        "Survival Ratio", 
        f"Survival Ratio vs {self.parameter_name}", 
        f"survival_ratio_vs_{self.parameter_name}.png")

    def plot_survival_per_atom(self, atom_index:int):
        survival_ratios = []
        initial = (self.images_data["image_type"] == 0)
        final = (self.images_data["image_type"] == 1)
        for param_idx in range(self.n_parameters):    
            mask = (self.images_data["parameter_index"] == param_idx)
            ref_atom_presence = 0
            final_atom_presence = 0
            for rep_idx in range(self.n_reps):
                ref_occupancy = self.images_data[mask & initial & (self.images_data["repetition_index"] == rep_idx)]["atoms_in_image"].tolist()
                final_occupancy = self.images_data[mask & final & (self.images_data["repetition_index"] == rep_idx)]["atoms_in_image"].tolist()
                if ref_occupancy[0][atom_index] == 1:
                    ref_atom_presence += 1
                if final_occupancy[0][atom_index] == 1:
                    final_atom_presence += 1
            survival_ratios.append(final_atom_presence / ref_atom_presence if ref_atom_presence > 0 else None)
        self.plot_graph(self.parameters, 
        survival_ratios, 
        self.parameter_name, 
        "Survival Ratio", 
        f"Survival Ratio vs {self.parameter_name}", 
        f"survival_ratio_vs_{self.parameter_name}_atom_{atom_index}.png")

        # self.plot_graph(self.parameters, 
        # survival_ratios, 
        # self.parameter_name, 
        # "Survival Ratio", 
        # f"Survival Ratio vs {self.parameter_name}", 
        # f"survival_ratio_vs_{self.parameter_name}_atom_{atom_index}.png")

    def atoms_in_image(self, image_path:str) -> np.ndarray:
        occupancy_matrix=[]
        image = Image.open(os.path.join(self.images_folder_path, image_path))
        pixel_values = np.array(image)
        pixel_values = pixel_values.astype(np.float32)
        for atom in self.atoms:
            if pixel_values[atom["position_y"]-self.roi_side_length//2:atom["position_y"]-self.roi_side_length//2+self.roi_side_length, atom["position_x"]-self.roi_side_length//2:atom["position_x"]-self.roi_side_length//2+self.roi_side_length].sum() > atom["threshold"]:
                occupancy_matrix.append(1)
            else:
                occupancy_matrix.append(0)
        # print(f"atom count in {image_path} is {atom_count}")
        return np.array(occupancy_matrix)

    def atoms_survival_ratio(self, ref_occupancy:np.ndarray, final_occupancy:np.ndarray): 
        total_final = final_occupancy.sum()
        total_ref = ref_occupancy.sum()
        return total_final / total_ref if total_ref > 0 else None
    
def main():
    images_folder_path ="/home/ohmorig-neo/image_analyzer/Jan09_2026_180036first_mw_resonance"
    scan_order = "params first"
    n_reps = 10
    n_parameters = 80
    images_per_cycle = 2
    parameters = np.linspace(0, 1, n_parameters)
    parameter_name = "mw_frequency"
    analyzer = ImageAnalyzer(calibration_file="calibration.json",
    images_folder_path=images_folder_path,
    scan_order=scan_order,
    n_reps=n_reps,
    n_parameters=n_parameters,
    images_per_cycle=images_per_cycle,
    parameters=parameters,
    parameter_name=parameter_name
    )

    analyzer.analyze_images()
    # analyzer.plot_survival_vs_parameter()
    analyzer.plot_survival_per_atom(43)

if __name__ == "__main__":
    main()