import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from skimage.feature import peak_local_max
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import json
from datetime import datetime
class ImageCalibrator:
    def __init__(self,grid_size:tuple[int, int] = (10, 10), roi_side_length:int = 5, images_folder_path:str = None):
        self.grid_size = grid_size
        self.roi_side_length = roi_side_length
        self.images_folder_path = images_folder_path

    def get_averaged_image(self, save_averaged_image:bool = False):
        averaged_image = np.zeros((256,256),dtype=np.float32)
        for image_path in os.listdir(self.images_folder_path):
            if image_path.endswith(".tif"):
                image = Image.open(os.path.join(self.images_folder_path, image_path))
                pixel_values = np.array(image)
                pixel_values = pixel_values.astype(np.float32)
                pixel_values /= pixel_values.max()
                averaged_image += pixel_values
        averaged_image /= averaged_image.max()
        if save_averaged_image:
            plt.imshow(averaged_image)
            total_images = len(os.listdir(self.images_folder_path))
            plt.title(f'Averaged Image of {total_images} images')
            plt.savefig('averaged_image.png')
        self.averaged_image = averaged_image
    
    def get_peak_coordinates(self,plot_rois:bool = False):
        averaged_image = self.averaged_image
        peak_coordinates = peak_local_max(averaged_image, min_distance=10)
        peaks_to_delete = []
        region_of_interest = []
        for index, coordinate in enumerate(peak_coordinates):
            # circle = patches.Circle(coordinate, self.roi_side_length/2, color='red', fill=False)
            # plt.gca().add_patch(circle)
            if averaged_image[coordinate[0], coordinate[1]] < 0.7:
                peaks_to_delete.append(index)
                continue
        peak_coordinates = np.delete(peak_coordinates, peaks_to_delete, axis=0)
        print(f"Found {len(peak_coordinates)} peaks")

        sort_x_index = np.argsort(peak_coordinates[:, 0])
        peak_coordinates = peak_coordinates[sort_x_index]
        if plot_rois:
            for index, coordinate in enumerate(peak_coordinates):
                plt.gca().add_patch(patches.Rectangle((coordinate[1]-self.roi_side_length/2, coordinate[0]-self.roi_side_length/2), self.roi_side_length, self.roi_side_length, color='red', fill=False))
                plt.gca().text(coordinate[1], coordinate[0], f'{index}', color='red')
        
            plt.title(f'Peak Coordinates and ROIs')
        self.peak_coordinates = peak_coordinates
    
    def photon_count_histogram_per_atom(self, atom_index:int, save_histogram:bool = False):
        """returns a list that contains the photon counts in the region of interest of the given atom index for all images"""
        position = self.peak_coordinates[atom_index]
        photon_counts = []
        images = sorted(os.listdir(self.images_folder_path))
        for image_path in images:
            if image_path.endswith(".tif"):
                image = Image.open(os.path.join(self.images_folder_path, image_path))
                pixel_values = np.array(image)
                pixel_values = pixel_values.astype(np.float32)
                # pixel_values /= pixel_values.max()
                photon_counts.append(pixel_values[position[0]-self.roi_side_length//2:position[0]-self.roi_side_length//2+self.roi_side_length, position[1]-self.roi_side_length//2:position[1]-self.roi_side_length//2+self.roi_side_length].sum())
        if save_histogram:
            plt.hist(photon_counts, bins=100)
            plt.xlabel('Photon Count')
            plt.ylabel('Frequency')
            # plt.imshow(roi)
            plt.title(f'Photon Count Histogram per Atom {atom_index}')
            plt.savefig(f'photon_count_histogram_per_atom_{atom_index}.png')
        return photon_counts
    
    def photon_count_histogram_average(self,save_histogram:bool = False):
        avreaged_photon_counts = np.zeros(self.grid_size[0]*self.grid_size[1])
        for atom_index in range(self.grid_size[0]*self.grid_size[1]):
            photon_counts = self.photon_count_histogram_per_atom(atom_index)
            avreaged_photon_counts += photon_counts
        avreaged_photon_counts /= self.grid_size[0]*self.grid_size[1]
        print(len(avreaged_photon_counts))
        if save_histogram:
            plt.hist(avreaged_photon_counts, bins=100)
            plt.xlabel('Photon Count')
            plt.ylabel('Frequency')
            plt.title(f'Photon Count Histogram Average')
            plt.savefig('photon_count_histogram_average.png')
        return avreaged_photon_counts

    def get_threshold(self, photon_counts: np.ndarray, plot_fit: bool = False):
        """Fit a bimodal Gaussian to photon counts and extract threshold between peaks."""
        
        def bimodal_gaussian(x, a1, mu1, sigma1, a2, mu2, sigma2):
            """Sum of two Gaussian distributions."""
            g1 = a1 * np.exp(-((x - mu1) ** 2) / (2 * sigma1 ** 2))
            g2 = a2 * np.exp(-((x - mu2) ** 2) / (2 * sigma2 ** 2))
            return g1 + g2
        
        # Create histogram
        counts, bin_edges = np.histogram(photon_counts, bins=100)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Find initial guesses for peaks
        peaks, _ = find_peaks(counts, height=np.max(counts) * 0.1, distance=10)
        
        if len(peaks) >= 2:
            # Sort peaks by bin_centers value to get low and high peaks
            peak_positions = bin_centers[peaks]
            sorted_indices = np.argsort(peak_positions)
            peak1_idx, peak2_idx = peaks[sorted_indices[0]], peaks[sorted_indices[-1]]
            mu1_init, mu2_init = bin_centers[peak1_idx], bin_centers[peak2_idx]
            a1_init, a2_init = counts[peak1_idx], counts[peak2_idx]
        else:
            # Fallback: use quartiles for initial guesses
            mu1_init = np.percentile(photon_counts, 25)
            mu2_init = np.percentile(photon_counts, 75)
            a1_init = a2_init = np.max(counts) / 2
        
        sigma_init = (np.max(photon_counts) - np.min(photon_counts)) / 10
        
        # Initial parameters: [a1, mu1, sigma1, a2, mu2, sigma2]
        p0 = [a1_init, mu1_init, sigma_init, a2_init, mu2_init, sigma_init]
        
        # Bounds to keep parameters reasonable
        bounds = (
            [0, np.min(photon_counts), 0, 0, np.min(photon_counts), 0],
            [np.inf, np.max(photon_counts), np.inf, np.inf, np.max(photon_counts), np.inf]
        )
        
        # Fit bimodal Gaussian
        popt, _ = curve_fit(bimodal_gaussian, bin_centers, counts, p0=p0, bounds=bounds, maxfev=10000)
        a1, mu1, sigma1, a2, mu2, sigma2 = popt
        
        # Ensure mu1 < mu2 (low peak first)
        if mu1 > mu2:
            a1, mu1, sigma1, a2, mu2, sigma2 = a2, mu2, sigma2, a1, mu1, sigma1
        
        # Find threshold: minimum of fitted curve between the two means
        x_between = np.linspace(mu1, mu2, 1000)
        y_between = bimodal_gaussian(x_between, a1, mu1, sigma1, a2, mu2, sigma2)
        threshold = x_between[np.argmin(y_between)]
        
        if plot_fit:
            plt.figure()
            plt.hist(photon_counts, bins=100, alpha=0.7, label='Data')
            x_fit = np.linspace(np.min(photon_counts), np.max(photon_counts), 500)
            plt.plot(x_fit, bimodal_gaussian(x_fit, *popt), 'r-', linewidth=2, label='Bimodal Gaussian Fit')
            plt.axvline(threshold, color='g', linestyle='--', linewidth=2, label=f'Threshold = {threshold:.1f}')
            plt.xlabel('Photon Count')
            plt.ylabel('Frequency')
            plt.title('Bimodal Gaussian Fit with Threshold')
            plt.legend()
            plt.savefig('threshold_fit.png')
        
        return threshold
    
    def create_calibration_file(self):
        calibration_data= {
            "calibration_date": datetime.now().strftime("%Y-%m-%d"),
            "calibration_data": self.images_folder_path,
            "roi_side_length": self.roi_side_length,
            "grid_size":self.grid_size,
        }

        for atom_index in range(self.grid_size[0]*self.grid_size[1]):
            calibration_data[f"atom {atom_index}"] ={
                "threshold": self.get_threshold(self.photon_count_histogram_per_atom(atom_index), plot_fit=False),
                "position_y": int(self.peak_coordinates[atom_index][0]),
                "position_x": int(self.peak_coordinates[atom_index][1]),
            }

        
        with open('calibration.json', 'w') as f:
            json.dump(calibration_data, f)

    def calibrate(self):
        """Load TIF images, find positions, compute threshold."""
        self.get_averaged_image()
        self.get_peak_coordinates()
        # self.photon_count_histogram_per_atom(89)
        # self.photon_count_histogram_average(save_histogram=True)
        # self.get_threshold(self.photon_count_histogram_per_atom(89), plot_fit=True)
        # plt.savefig('atom_43_photon_count_histogram.png')
        self.create_calibration_file()





def main():
    images_folder_path = input("Enter the path to the images folder: ")
    images_folder_path = "./Jan11_2026_132241calibration_image"
    calibrator = ImageCalibrator(grid_size=(10, 10), roi_side_length=5,images_folder_path=images_folder_path)
    calibrator.calibrate()

if __name__ == "__main__":
    main()