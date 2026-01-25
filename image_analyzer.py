import json
import os
from typing import List
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit


def gaussian(x, a, c, s, offset):
    return a * np.exp(-(x - c)**2 / (2 * s**2)) + offset

def triple_gaussian(x, a1, c1, s1, a2, c2, s2, a3, c3, s3, offset):
    """Sum of 3 Gaussian functions with a constant offset.
    
    Parameters:
        x: independent variable
        a1, a2, a3: amplitudes of each Gaussian
        c1, c2, c3: centers of each Gaussian
        s1, s2, s3: standard deviations (widths) of each Gaussian
        offset: constant baseline offset
    """
    g1 = a1 * np.exp(-(x - c1)**2 / (2 * s1**2))
    g2 = a2 * np.exp(-(x - c2)**2 / (2 * s2**2))
    g3 = a3 * np.exp(-(x - c3)**2 / (2 * s3**2))
    return g1 + g2 + g3 + offset

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
        # plt.savefig(save_path)
        plt.show()
    
    def plot_survival_vs_parameter(self, fit=True, initial_guess=None, model_function=triple_gaussian):
        """Plot survival ratio vs parameter with optional 3-Gaussian fit.
        
        Parameters:
            fit: bool, whether to perform and plot the fit (default True)
            initial_guess: list of 10 values [a1, c1, s1, a2, c2, s2, a3, c3, s3, offset]
                          If None, automatic guesses will be generated.
        
        Returns:
            dict with 'params' and 'errors' if fit=True and successful, None otherwise
        """
        survival_ratios = []
        ref_occupancies = []
        final_occupancies = []
        for param_idx in range(self.n_parameters):
            mask = (self.images_data["parameter_index"] == param_idx)
            ref_occupancy = self.images_data[mask & (self.images_data["image_type"] == 0)]["atoms_in_image"].sum()
            final_occupancy = self.images_data[mask & (self.images_data["image_type"] == 1)]["atoms_in_image"].sum()
            survival_ratio = self.atoms_survival_ratio(ref_occupancy, final_occupancy)
            survival_ratios.append(survival_ratio)
            ref_occupancies.append(ref_occupancy.sum())
            final_occupancies.append(final_occupancy.sum())
        self.survival_ratios = survival_ratios

        x_data = np.array(self.parameters)
        y_data = np.array(survival_ratios)
        ref = np.array(ref_occupancies)/max(ref_occupancies)
        final = np.array(final_occupancies)/max(final_occupancies)
        plt.figure()
        # plt.plot(x_data,ref,marker='o', label='Ref Occupancy')
        # plt.plot(x_data,final,marker='s', label='Final Occupancy')
        plt.plot(x_data, y_data, marker='^', label='ratio')
        
        fit_result = None
        
        if fit:
            # Filter out None values for fitting
            valid_mask = np.array([y is not None for y in y_data])
            x_fit_data = x_data[valid_mask]
            y_fit_data = np.array([y for y in y_data if y is not None])
            
            if initial_guess is None:
                # Generate automatic initial guesses
                x_range = x_fit_data.max() - x_fit_data.min()
                y_min, y_max = y_fit_data.min(), y_fit_data.max()
                amplitude_guess = (y_max - y_min) / 3
                width_guess = x_range / 10
                
                # Distribute centers across the data range
                c1_guess = x_fit_data.min() + x_range * 0.25
                c2_guess = x_fit_data.min() + x_range * 0.5
                c3_guess = x_fit_data.min() + x_range * 0.75
                
                initial_guess = [
                    amplitude_guess, c1_guess, width_guess,
                    amplitude_guess, c2_guess, width_guess,
                    amplitude_guess, c3_guess, width_guess,
                    y_min
                ]
            
            try:
                popt, pcov = curve_fit(model_function, x_fit_data, y_fit_data,initial_guess, maxfev=10000)
                perr = np.sqrt(np.diag(pcov))
                
                # Store fit results
                self.fit_params = popt
                self.fit_errors = perr
                fit_result = {'params': popt, 'errors': perr}
                
                # Generate smooth fit curve
                x_smooth = np.linspace(x_fit_data.min(), x_fit_data.max(), 500)
                y_fit = model_function(x_smooth, *popt)
                
                plt.plot(x_smooth, y_fit, 'r-', label='3-Gaussian Fit', linewidth=2)
                
                # Print fit results
                print("\n=== Fit Results ===")
                print(f"Gaussian 1: center = ({popt[1]:.4f} ± {perr[1]:.4f}), "
                      f"amplitude = {popt[0]:.4f} ± {perr[0]:.4f}, "
                      f"sigma = {popt[2]:.2f} ± {perr[2]:.2f}")
                print(f"Gaussian 2: center = ({popt[4]/1e6:.4f} ± {perr[4]/1e6:.4f}) × 10⁶, "
                      f"amplitude = {popt[3]:.4f} ± {perr[3]:.4f}, "
                      f"sigma = {popt[5]:.2f} ± {perr[5]:.2f}")
                print(f"Gaussian 3: center = ({popt[7]/1e6:.4f} ± {perr[7]/1e6:.4f}) × 10⁶, "
                      f"amplitude = {popt[6]:.4f} ± {perr[6]:.4f}, "
                      f"sigma = {popt[8]:.2f} ± {perr[8]:.2f}")
                print(f"Offset: {popt[9]:.4f} ± {perr[9]:.4f}")
                
            except RuntimeError as e:
                print(f"Fit failed: {e}")
                print("Try providing better initial_guess values.")
        
        plt.xlabel(self.parameter_name)
        plt.ylabel("Survival Ratio")
        plt.title(f"Survival Ratio vs {self.parameter_name}")
        plt.legend()
        # plt.savefig(f"survival_ratio_vs_{self.parameter_name}.png")
        plt.show()
        
        return fit_result

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