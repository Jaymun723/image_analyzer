import matplotlib.pyplot as plt
import numpy as np
from image_analyzer import ExperimentResult
from pprint import pprint as pp



class Visualizer:
    def __init__(self, data: ExperimentResult):
        self.data = data.data
        self.parameter_name = data.parameter_name
        self.images_per_cycle = data.images_per_cycle
        self.total_sites = data.total_sites
        self.fig, self.ax = plt.subplots()

    def clean_plot(self):
        self.ax.clear()

    def show_plot(self):
        self.fig.show()

    def plot_data(self):
        if self.images_per_cycle == 2:
            n = self.total_sites
            initial_atoms_fraction = self.data["total_atoms_initial"] / n
            final_atoms_fraction = self.data["total_atoms_final"] / n
            initial_se = np.sqrt(initial_atoms_fraction * (1 - initial_atoms_fraction) / n)
            final_se = np.sqrt(final_atoms_fraction * (1 - final_atoms_fraction) / n)
            self.ax.errorbar(self.data["parameter"], initial_atoms_fraction, yerr=initial_se, marker='o', label='Initial', color='blue',capsize=3)
            self.ax.errorbar(self.data["parameter"], final_atoms_fraction, yerr=final_se, marker='o', label='Final', color='red',capsize=3)
            self.ax.set_xlabel(self.parameter_name)
            self.ax.set_ylabel("atoms fraction")
            self.ax.set_title(f"{self.parameter_name} vs atoms fraction")
            self.ax.legend()
            self.fig.show(f"{self.parameter_name}_vs_atoms_fraction.png")
        else:
            total_atoms_fraction = self.data["total_atoms"] / n
            total_se = np.sqrt(total_atoms_fraction * (1 - total_atoms_fraction) / n)
            self.ax.errorbar(self.data["parameter"], total_atoms_fraction, yerr=total_se, marker='o', label='atoms fraction',capsize=3)
            self.ax.set_xlabel(self.parameter_name)
            self.ax.set_ylabel("atoms fraction")
            self.ax.set_title(f"{self.parameter_name} vs atoms fraction")
            self.ax.legend()
            self.fig.show(f"{self.parameter_name}_vs_atoms_fraction.png")
    
    def plot_survival_ratio(self):

        if self.images_per_cycle == 2:
            n = self.total_sites
            mask = (self.data["total_atoms_initial"] > 0)
            initial_atoms_fraction = self.data[mask]["total_atoms_initial"]
            final_atoms_fraction = self.data[mask]["total_atoms_final"]

            ratio = final_atoms_fraction / initial_atoms_fraction
            se = np.sqrt(ratio * (1 - ratio) / n)

            self.ax.errorbar(self.data["parameter"], ratio, yerr=se, marker='o', label='survival ratio',capsize=3)
            self.ax.set_xlabel(self.parameter_name)
            self.ax.set_ylabel("survival ratio")
            self.ax.set_title(f"{self.parameter_name} vs survival ratio")
            self.ax.legend()
            self.fig.show(f"{self.parameter_name}_vs_survival_ratio.png")
        else:
            pp("there is only one image per cycle")
