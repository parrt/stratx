import os

# Updated to shap-0.36.0, scikit-learn 0.23.1

# uncomment to re-tune models for all data sets;
# otherwise support.models dict (previously-computed defaults) are used
# import tuning

# generate CSV read by support code to guarantee separate train/valid sets
# and that they are the same across tuning, testing, etc...
import gen_valid_sets

# By importing these, the main program runs and generates images

import boston_various_models_shap
import bulldozer_YearMade_pdp
import bulldozer_YearMade_pdp_shap
import bulldozer_fdp_shap_imp
import bulldozer_top
# import compare_friedman_shap_stratimpact # I have cut this out of paper for now

import flights_top
import rent_top
import rent_stability
import rent_fdp_shap_imp

import synthetic_stability

# warning next one is slow:
# import speed
