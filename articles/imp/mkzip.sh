SRC=~/github/stratx/articles/imp
TARGET=/tmp/pdimp
IMG_TARGET=/tmp/pdimp/images
#IMG_TARGET=/tmp/pdimp

rm -rf $TARGET
rm /tmp/pdimp.zip
mkdir $TARGET
mkdir $IMG_TARGET
cd $SRC

cp pdimp.tex pdimp.bib pdimp.bbl preprint-svjour3.cls $TARGET

cp images/diff-models.pdf  $IMG_TARGET
cp images/quadratic-auc.pdf  $IMG_TARGET
cp images/bulldozer-YearMade.pdf  $IMG_TARGET
cp images/FPD-SHAP-PD.pdf  $IMG_TARGET
cp images/rent-pdp-vs-shap.pdf  $IMG_TARGET
cp images/bulldozer-pdp-vs-shap.pdf  $IMG_TARGET
cp images/boston-topk-RF-baseline.pdf  $IMG_TARGET
cp images/flights-topk-RF-baseline.pdf  $IMG_TARGET
cp images/bulldozer-topk-RF-baseline.pdf  $IMG_TARGET
cp images/rent-topk-RF-baseline.pdf  $IMG_TARGET
cp images/boston-topk-RF-Importance.pdf  $IMG_TARGET
cp images/flights-topk-RF-Importance.pdf  $IMG_TARGET
cp images/bulldozer-topk-RF-Importance.pdf  $IMG_TARGET
cp images/rent-topk-RF-Importance.pdf  $IMG_TARGET
cp images/boston-topk-RF-Impact.pdf  $IMG_TARGET
cp images/flights-topk-RF-Impact.pdf  $IMG_TARGET
cp images/bulldozer-topk-RF-Impact.pdf  $IMG_TARGET
cp images/rent-topk-RF-Impact.pdf  $IMG_TARGET
cp images/boston-topk-RF-Impact.pdf  $IMG_TARGET
cp images/flights-topk-RF-Impact.pdf  $IMG_TARGET
cp images/bulldozer-topk-RF-Impact.pdf  $IMG_TARGET
cp images/rent-topk-RF-Impact.pdf  $IMG_TARGET
cp images/boston-topk-GBM-Importance.pdf  $IMG_TARGET
cp images/flights-topk-GBM-Importance.pdf  $IMG_TARGET
cp images/bulldozer-topk-GBM-Importance.pdf  $IMG_TARGET
cp images/rent-topk-GBM-Importance.pdf  $IMG_TARGET
cp images/boston-topk-OLS-Importance.pdf  $IMG_TARGET
cp images/flights-topk-OLS-Importance.pdf  $IMG_TARGET
cp images/rent-topk-OLS-Importance.pdf  $IMG_TARGET
cp images/rent-stability-importance.pdf  $IMG_TARGET
cp images/rent-stability-impact.pdf  $IMG_TARGET
cp images/boston-features.pdf  $IMG_TARGET
cp images/boston-features-shap-rf.pdf  $IMG_TARGET
cp images/flights-features.pdf  $IMG_TARGET
cp images/flights-features-shap-rf.pdf  $IMG_TARGET
cp images/bulldozer-features.pdf  $IMG_TARGET
cp images/bulldozer-features-shap-rf.pdf  $IMG_TARGET
cp images/rent-features.pdf  $IMG_TARGET
cp images/rent-features-shap-rf.pdf  $IMG_TARGET
cp images/*Linear*.pdf  $IMG_TARGET
cd $TARGET
zip -r ../pdimp.zip *
