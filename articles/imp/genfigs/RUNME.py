import os

# generate CSV read by R code
import gencsv

# uncomment to re-tune models for all data sets;
# otherwise support.models dict defaults are used
# import tuning

# Invoke R to generate csv files then load with python to plot
# Exec R and generate images/*.csv files.  Then plot with Python
os.system("R CMD BATCH ale_plots_bulldozer.R")

# By importing these, the main program runs and generates images

import boston_various_models_shap
import bulldozer_YearMade_pdp
import bulldozer_YearMade_pdp_shap
import bulldozer_pdp_shap_imp
import compare_friedman_shap_stratimpact

import flights_top
import rent_top
import rent_stability
import rent_pdp_shap_imp
import bulldozer_top