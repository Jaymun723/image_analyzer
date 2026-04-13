import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from src.image_analyzer import ExperimentResult
from pprint import pprint as pp
from typing import Callable, Optional, Tuple


class Visualizer:
    def __init__(self, data: ExperimentResult):
        self.data = data.data
        self.parameter_name = data.parameter_name
        self.images_per_cycle = data.images_per_cycle
        self.total_sites = data.total_sites
        self.fig, self.ax = plt.subplots()
        # Last plotted data, used by fit_curve() when x/y not provided
        self._last_x: Optional[np.ndarray] = None
        self._last_y: Optional[np.ndarray] = None
        self._last_yerr: Optional[np.ndarray] = None

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
            self.fig.canvas.draw()
            self._last_x = np.asarray(self.data["parameter"])
            self._last_y = initial_atoms_fraction.values
            self._last_yerr = initial_se.values
        else:
            n = self.total_sites
            total_atoms_fraction = self.data["total_atoms"] / n
            total_se = np.sqrt(total_atoms_fraction * (1 - total_atoms_fraction) / n)
            self.ax.errorbar(self.data["parameter"], total_atoms_fraction, yerr=total_se, marker='o', label='atoms fraction',capsize=3)
            self.ax.set_xlabel(self.parameter_name)
            self.ax.set_ylabel("atoms fraction")
            self.ax.set_title(f"{self.parameter_name} vs atoms fraction")
            self.ax.legend()
            self.fig.canvas.draw()
            self._last_x = np.asarray(self.data["parameter"])
            self._last_y = total_atoms_fraction.values
            self._last_yerr = total_se.values

    def plot_survival_ratio(self):

        if self.images_per_cycle == 2:
            n = self.total_sites
            mask = (self.data["total_atoms_initial"] > 0)
            initial_atoms_fraction = self.data[mask]["total_atoms_initial"]
            final_atoms_fraction = self.data[mask]["total_atoms_final"]

            ratio = final_atoms_fraction / initial_atoms_fraction
            se = np.sqrt(ratio * (1 - ratio) / n)

            x_plot = self.data.loc[mask, "parameter"].values
            self.ax.errorbar(x_plot, ratio.values, yerr=se.values, marker='o', label='survival ratio', capsize=3)
            self.ax.set_xlabel(self.parameter_name)
            self.ax.set_ylabel("survival ratio")
            self.ax.set_title(f"{self.parameter_name} vs survival ratio")
            self.ax.legend()
            self.fig.canvas.draw()
            self._last_x = np.asarray(x_plot)
            self._last_y = ratio.values
            self._last_yerr = se.values
        else:
            pp("there is only one image per cycle")

    def fit_curve(
        self,
        func: Callable[..., float],
        x: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        p0: Optional[Tuple[float, ...]] = None,
        sigma: Optional[np.ndarray] = None,
        label: str = "fit",
        param_names: Optional[Tuple[str, ...]] = None,
        show_params: bool = True,
        **curve_fit_kw,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit a function to the plotted data, draw the fit curve, and show fitted parameters.

        func: callable f(x, *args) -> y
        x, y: data to fit. If None, use the last data plotted by plot_data or plot_survival_ratio.
        p0: initial guess for parameters (optional).
        sigma: uncertainties in y for weighted fit (e.g. 1/sigma from _last_yerr).
        label: legend label for the fit curve.
        param_names: names for parameters in the text box (e.g. ("A", "τ", "c")). If None, use p0, p1, ...
        show_params: if True, display fitted parameters and uncertainties in a text box on the plot.
        **curve_fit_kw: passed to scipy.optimize.curve_fit.

        Returns:
            popt: fitted parameters
            pcov: covariance matrix
        """
        x_fit = np.asarray(x) if x is not None else self._last_x
        y_fit = np.asarray(y) if y is not None else self._last_y
        if x_fit is None or y_fit is None:
            raise ValueError("No data to fit. Call plot_data() or plot_survival_ratio() first, or pass x and y.")
        sigma_fit = sigma
        if sigma_fit is None and self._last_yerr is not None:
            sigma_fit = np.asarray(self._last_yerr)
        popt, pcov = curve_fit(func, x_fit, y_fit, p0=p0, sigma=sigma_fit, absolute_sigma=(sigma_fit is not None), **curve_fit_kw)
        perr = np.sqrt(np.diag(pcov))

        x_smooth = np.linspace(x_fit.min(), x_fit.max(), 200)
        self.ax.plot(x_smooth, func(x_smooth, *popt), "--", label=label)

        if show_params:
            names = param_names or tuple(f"p{i}" for i in range(len(popt)))
            lines = [f"{name} = {v:.4g} ± {e:.4g}" for name, v, e in zip(names, popt, perr)]
            text = "\n".join(lines)
            self.ax.text(0.02, 0.98, text, transform=self.ax.transAxes, fontsize=9,
                        verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

        self.ax.legend()
        self.fig.canvas.draw()
        return popt, pcov

