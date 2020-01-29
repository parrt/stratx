# Notes and checklists

* min val per cat option like min slopes per x? outliers? didn't help ModelID
* add ignored to plots
* yearmade and modelid get reversed with density_weighted; see compute_importance(); pvalue=1.0?
  Seems that modelid with shuffled y is random and large
* if mean - 2sigma <=0 then can discard feature
* timing experiment to see empirical complexity
* plots can't handle unnormalized data
* plots show rank then importance? option to sortby
* use same splits for all CV; seems to wobble a bit between top-k curves even on same first var