import os

# generate CSV read by support code to guarantee separate train/valid sets
# and that they are the same across tuning, testing, etc...
import gen_valid_sets

# uncomment to re-tune models for all data sets;
# otherwise support.models dict defaults are used
# import tuning

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