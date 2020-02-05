# Notes and checklists

* min val per cat option like min slopes per x? outliers? didn't help ModelID
* add ignored to plots
* yearmade and modelid get reversed with density_weighted; see compute_importance(); pvalue=1.0?
  Seems that modelid with shuffled y is random and large
* timing experiment to see empirical complexity
* plots can't handle unnormalized data
* plots show rank then importance? option to sortby
* feature plot doesn't show error bars
* set range on top-k plots to be same for same dataset
* errors running model_free_baseline; not sure it's my stuff, likely not Strat stuff
```
invalid value encountered in true_divide
invalid value encountered in true_divide
invalid value encountered in greater
invalid value encountered in less
invalid value encountered in less_equal
```
* don't say "top-4 training ..." in R dataframe
* don't compress X[colname] in plot catstrat; messes up labels
* set default rank to Importance or Impact?
* try X_train=X for computing features again?
* error bars are missing