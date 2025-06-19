# brain-age-prediction

This repo contains two scripts:

| Script | Description |
|--------|-------------|
| **training_validation.R** | Trains a model to predict brain age from imaging features |
| **predict_test_set.R**  | Loads a saved model and evaluates it or predicts on new data |

These scripts are adapted from code used for brain age testing and training used in \
Beck et al. (https://doi.org/10.1016/j.biopsych.2024.07.019) bar slight changes in the following: \

- 5-fold x 2-repeat cross validation instead of 10-fold to cut run time 80% and using parallel back-end so it utilizes all cores. \
- Leak-free preprocessing – NZV and normalization are now estimated inside each resample \
so hyper-parameter tuning sees slightly “truer” error and picks better settings. \
- Larger effective training set per tree – because we dropped repeated hand-baking, \
more features survived NZV in some folds, giving the booster a richer signal. \
- Identifiers (src_subject_id, eventname) are kept but given role = "id" and ignored by XGBoost to prevent over-fit to person or visit labels.
- Joint tuning of trees and learn_rate – the model can trade depth for shrinkage; \
final parameters usually give a marginal accuracy gain. \

## Data preparation

Prepare your data files into two data frames; one for the training (and validation) sample, and one for the hold-out test sample. \
In my data preparation, I make a Sample1.Rda (loaded in the first script), and a Sample2.Rda (loaded in second script). \
These two data frames represent a 50:50 cohort split using the ABCD Study data (baseline and two-year follow-up - release 5.1) \
following QC and longCombat harmonization of imaging data.

The 50:50 split includes a subject-wise split across time-points that ensures baseline and follow-up scans from the same participant \
remain in the same partition, avoiding identify-confounding. Siblings are dealt with using a group shuffle split \
with family ID as the group indicator to ensure that no siblings were split across training and test sets. \
Sex that is not equal to 1 or 2 is removed. The resulting two data frames (sample1 and sample2) represent a final N that is identical \ 
(or difference of 1) and has as equal as possible distribution of age range, sex split, and cross-sectional versus longitudinal data points.

The scripts below use T1-weighted imaging data but scripts are also available for dMRI, rs-fMRI, and a multi-modal model.\


## Instructions

## 1. Training
Rscript train_brain_age.R \

In this script, Sample1.Rda is loaded as the training sample following data preparation steps outlined above. \
A training/validation split at 80:20 is made (meaning the overarching sample split for training/validation/testing is 40/10/50 percent). \


## 2. Test / predict
Rscript test_brain_age.R \

In this script, Sample2.Rda (hold-out sample) is loaded as the test set. \
Age-bias correction is carried out using correction procedures outlined in de Lange & Cole (https://doi.org/10.1016/j.nicl.2020.102229)


### Contact
If you have any questions about the code or wish to collaborate, please contact me at [dani.beck@psykologi.uio.no](mailto:dani.beck@psykologi.uio)




