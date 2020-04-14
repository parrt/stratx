import os

# Invoke R to generate csv files then load with python to plot

"Exec R and generate images/*.csv files.  Then plot with Python"
os.system("R CMD BATCH ale_plots_bulldozer.R")

# By importing these, the main program runs and generates images

import boston_various_models_shap
import bulldozer_YearMade_pdp
import compare_friedman_shap_stratimpact