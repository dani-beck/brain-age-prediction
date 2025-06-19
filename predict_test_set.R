# ───────────────────────────────────────────────────────────────────────────
# 1.  Packages and working directory 
# ───────────────────────────────────────────────────────────────────────────

library(tidymodels)
library(xgboost)

# set wd
setwd("/tsd/p33/scratch/no-backup/users/daniabe/3peat_brainage/t1_brainAGE")

# ───────────────────────────────────────────────────────────────────────────
# 2.  Load fitted artefacts produced by the training script
# ───────────────────────────────────────────────────────────────────────────

load("fit_workflow.rds")   # brings object `fit_workflow`
fit_workflow

load("final_xgb.rds")      # object `final_xgb` (optional, spec only)
final_xgb

# ───────────────────────────────────────────────────────────────────────────
# 3.  Load hold-out sample (50 % split)
# ───────────────────────────────────────────────────────────────────────────

load("Sample2.Rda")

# ───────────────────────────────────────────────────────────────────────────
# 4.  Predict brain age in the hold-out cohort (Sample 2)
# ───────────────────────────────────────────────────────────────────────────

# Predict & assemble brain-age data frame with extra metadata
brain_age_pred <- predict(fit_workflow, new_data = sample_2) %>%
  # add identifiers and metadata in one go
  dplyr::bind_cols(
    sample_2 %>% 
      dplyr::select(src_subject_id,      # participant ID
                    eventname,           # time-point / visit label
                    sex,                 # biological sex
                    interview_age)       # chronological age
  ) %>%
  dplyr::rename(truth = interview_age) %>%
  dplyr::mutate(gap = .pred - truth) %>%
  
  # tidy column order: ID first, then metadata, then predictions
  dplyr::select(src_subject_id, eventname, sex,
                truth, .pred, gap)

# Performance check on the hold-out sample
brain_age_pred %>%
  yardstick::metrics(truth = truth, estimate = .pred)


# ───────────────────────────────────────────────────────────────────────────
# 5.  Age-bias correction based on the validation (20 %) split
# ───────────────────────────────────────────────────────────────────────────

load("df_validation.rds")

val_pred <- predict(fit_workflow, new_data = df_validation) %>%
  dplyr::mutate(truth = df_validation$interview_age)

bias_mod       <- lm(.pred ~ truth, data = val_pred)
bias_intercept <- coef(bias_mod)[1]
bias_slope     <- coef(bias_mod)[2]

# Apply correction to hold-out predictions
brain_age_pred <- brain_age_pred %>%
  dplyr::mutate(
    corrected_pred = (.pred - bias_intercept) / bias_slope,
    corrected_gap  = corrected_pred - truth
  ) %>%
  # final column order
  dplyr::select(src_subject_id, eventname, sex,
                truth, .pred, corrected_pred,
                gap,   corrected_gap)


# ───────────────────────────────────────────────────────────────────────────
# 6.  Save results
# ───────────────────────────────────────────────────────────────────────────

save(brain_age_pred, file = "abcd_5.1_T1_brain_age.rds")


# ───────────────────────────────────────────────────────────────────────────
# 7.  Sanity checks you can run 
# ───────────────────────────────────────────────────────────────────────────

# Correlation between corrected GAP and age should approach zero
brain_age_pred %>%
  summarise(r = cor(corrected_gap, truth))

# Plot raw vs corrected predictions
library(ggplot2)
ggplot(brain_age_pred, aes(truth, .pred)) +
  geom_point(alpha = .1) +
  geom_abline(lty = 2) +
  geom_abline(intercept = bias_intercept, slope = bias_slope, colour = "red") +
  labs(title = "Calibration: raw (red) vs ideal (black dashed)")

ggplot(brain_age_pred, aes(truth, corrected_pred)) +
  geom_point(alpha = .1) +
  geom_abline(lty = 2) +
  labs(title = "After bias correction")

# the rmse and mae slightly improved (0.03) from the previous model used in the ELA paper
# Leak-free preprocessing – NZV and normalisation are now estimated inside each resample, 
# so hyper-parameter tuning sees slightly “truer” error and picks better settings.
# 
# Joint tuning of trees and learn_rate – the model can trade depth for shrinkage; 
# final parameters usually give a marginal accuracy gain.
# 
# Larger effective training set per tree – because we dropped repeated hand-baking, 
# more features survived NZV in some folds, giving the booster a richer signal.
# 
# A ~0.03 y drop in both RMSE and MAE is typical when you remove information leakage 
# and fine-tune learning-rate/trees together—small but real.


