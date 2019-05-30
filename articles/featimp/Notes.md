# Feature importance

## Problems

Permutation or drop column importance:

* relies on accuracy / appropriateness of model
* need y; must be supervised
* specific to that model not the data
* use training or testing? unclear
* collinearity causes shared importance between features and 0 importance for dropcol method
* permutation causes nonsensical obs

## Ideas from stratification

If partial dependence is 0, importance must be zero as it has no effect.

standardized lin reg coeff's still can be biased due to multicolinearity.

## Advantages

simple idea: "amount of y variance explained by x_c with all else being equal"

* model indep (not dep on accurate model or diff per model)
* insensitive to codep
* simple
* can show how feat impo changes over x_c range

## Articles, related work

* [Permutation importance: a corrected feature importance measure](https://academic.oup.com/bioinformatics/article/26/10/1340/193348)
* [null importances](https://www.kaggle.com/ogrellier/feature-selection-with-null-importances)
* SHAP
* permutation imp
* drop col
* gini drop