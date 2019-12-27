# Notes about StratImpact

## Dec 15, 2019

Runs of lin eqn generated data with x3=x1+noise

OLS coeff [1. 1. 1.]
RF min_samples_leaf=3, backgroundsize= 300, 25 trials of 5000 records: [1.04868524 0.98700816 0.94670527] with stddev [0.26978524 0.00705834 0.26832021]
RF min_samples_leaf=3, backgroundsize= 400, 15 trials of 5000 records: [0.96985067 0.98977231 1.02502662] with stddev [0.31783823 0.00881368 0.31568062]
RF min_samples_leaf=3, backgroundsize= 500, 25 trials of 5000 records: [1.070995   0.98873199 0.92438001] with stddev [0.26462216 0.00629446 0.26298054]
RF min_samples_leaf=3, backgroundsize= 100, 25 trials of 5000 records: [0.97432242 0.98762778 1.01998368] with stddev [0.29143514 0.00511653 0.28907165]
RF min_samples_leaf=3, backgroundsize=1000, 25 trials of 1000 records: [0.91799695 0.95951468 1.06823948] with stddev [0.20212024 0.00599243 0.20096446]
RF min_samples_leaf=3, backgroundsize=1000, 50 trials of 1000 records: [0.95297787 0.96216446 1.03382709] with stddev [0.27070884 0.0062901  0.26944995]
RF min_samples_leaf=3, backgroundsize= 300, 50 trials of 2000 records: [0.99995283 0.97702764 0.9910488 ] with stddev [0.27109678 0.00751501 0.269343  ]
RF min_samples_leaf=3, backgroundsize= 300,100 trials of 2000 records: [0.99563874 0.97791343 0.99558951] with stddev [0.26213271 0.00748368 0.26038133]

StratImpact w/min_samples_leaf= 3, 15 trials of 5000 records: [0.7650521  0.99800097 0.76250869] with stddev [0.00996777 0.00113329 0.0115253 ]
StratImpact w/min_samples_leaf= 5, 15 trials of 5000 records: [0.83871061 0.99775181 0.8340384 ] with stddev [0.0104251  0.00049489 0.01870721]
StratImpact w/min_samples_leaf= 7, 15 trials of 5000 records: [0.88385418 0.99774893 0.8926547 ] with stddev [0.02056113 0.00090538 0.01369687]
StratImpact w/min_samples_leaf= 8, 15 trials of 5000 records: [0.92114239 0.99733657 0.91241468] with stddev [0.01426463 0.00071245 0.01463774]
StratImpact w/min_samples_leaf=10, 15 trials of 5000 records: [0.95981299 0.99727285 0.94669593] with stddev [0.0180184  0.00110054 0.01398092]
StratImpact w/min_samples_leaf=12, 15 trials of 5000 records: [0.99068302 0.99735248 0.98498582] with stddev [0.02276007 0.00063022 0.02374084]
StratImpact w/min_samples_leaf=14, 15 trials of 5000 records: [1.02976418 0.99641839 1.01149308] with stddev [0.02290239 0.00067753 0.0220821 ]

James pointed out [Cook's distance](https://en.wikipedia.org/wiki/Cook%27s_distance). Looks simple enough and "Cook's distance measures the effect of deleting a given observation."

## Dec 27, 2019

Use special loss function in RF to measure purity of X not y. We don't really want to squash y as we fit tree.  Use example of step function and good dtree fit that splits at discontinuity. That emphasizes many different x values, which is not what we want; want to group by x. Then show example where most pure y subregions gives wrong answer for most-pure-x-regions.

ISSUES for shap: 
* categoricals
* measure diff from avg output not from 0

Did interaction.py term experiment. shap and we get same partial dependences and feature weights.

added p-values via null distr recently.

main paper features:

* define feature imp as ratio of masses under partial dependence curve; partial deriv stuff
* we have model-free partial dep which allows model free imporantances
* linear complexity, paralleilizable.

do we need to talk about LIME and others that subsumed by SHAP?