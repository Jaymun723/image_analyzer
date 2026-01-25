from image_analyzer import ImageAnalyzer
import numpy as np

images_folder_path = "/home/ohmorig-neo/image_analyzer/Jan20_2026_170940stretch"

images_folder_path_lowtrap = "/home/ohmorig-neo/image_analyzer/Jan21_2026_134854stretchresonance_lowtrap_1v_tagrack"
images_folder_path_hightrap = "/home/ohmorig-neo/image_analyzer/Jan21_2026_140712stretchresonance_hightrap_4v_tagrack"

scan_order = "params first"
n_reps = 30
n_parameters = 35
images_per_cycle = 2
parameters = [8560110., 8570110., 8580110., 8590110., 8600110.,
8610110., 8622610., 8635110., 8647610., 8660110.,
8670110., 8682610., 8695110., 8707610., 8720110.,
8730110., 8740110., 8750110., 8760110., 8770110.,
8780110., 8790110., 8800110., 8810110., 8820110.,
8830110., 8840110., 8850110., 8860110., 8870110.,
8880110., 8890110., 8900110., 8910110., 8920110.]
parameter_name = "mw frequency (high trap)"
analyzer = ImageAnalyzer(calibration_file="calibration.json",images_folder_path=images_folder_path_hightrap,scan_order=scan_order,n_reps=n_reps,n_parameters=n_parameters,parameters=parameters,parameter_name=parameter_name, images_per_cycle=images_per_cycle)

analyzer.analyze_images()
analyzer.plot_survival_vs_parameter()