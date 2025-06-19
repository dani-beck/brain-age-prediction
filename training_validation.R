# ──────────────────────────────────────────────────────────────────────
# 1.  Packages & setup, load data
# ──────────────────────────────────────────────────────────────────────

library(dplyr)
library(doParallel)
library(tidymodels)
library(xgboost)

setwd("/tsd/p33/scratch/no-backup/users/daniabe/3peat_brainage/t1_brainAGE")

# load data 
load("Sample1.Rda")

# ──────────────────────────────────────────────────────────────────────
# 2.  Train / Validation split 80/20
# ──────────────────────────────────────────────────────────────────────

set.seed(42)
# Draw 80 % of participants (by ID) for the training partition
unique_ids <- unique(sample_1$src_subject_id)
train_ids <- sample(unique_ids, size = floor(0.8 * length(unique_ids)))

# Slice the data in a single step each
df_train      <- sample_1 %>% dplyr::filter(src_subject_id %in% train_ids)
df_validation <- sample_1 %>% dplyr::filter(!src_subject_id %in% train_ids)

# Save data frames to work directory
save(df_train, file = "df_train.rds")
save(df_validation, file = "df_validation.rds")

# Clean as you go
rm(sample_1, train_ids, unique_ids)


# ──────────────────────────────────────────────────────────────────────
# 3.  Pre-processing recipe (keeps IDs, drops sex, no early prep)
# ──────────────────────────────────────────────────────────────────────

brainage_recipe <- recipe(interview_age ~ ., data = df_train) %>%
  # Keep identifiers for bookkeeping but exclude from predictors
  update_role(src_subject_id, eventname, new_role = "id") %>%
  # Drop sex so the model relies purely on brain features
  step_rm(sex) %>%
  # Clean-up steps
  step_nzv(all_predictors()) %>%            # remove near-zero-variance cols
  step_normalize(all_numeric_predictors())  # optional, cheap, scale-invariant


# ──────────────────────────────────────────────────────────────────────
# 4.  XGBoost Model specification and hyper-parameter grid
# ──────────────────────────────────────────────────────────────────────

## 4.1 fully tunable trees
boost_mod <- boost_tree(
  mode           = "regression",
  trees          = tune(),                # 200 – 1200
  tree_depth     = tune(),                # 3 – 10
  min_n          = tune(),                # 2 – 40
  loss_reduction = tune(),                # γ
  sample_size    = tune(),                # 0.5 – 1.0
  mtry           = tune(),                # will map to colsample_bytree
  learn_rate     = tune()                 # 1e-5 – 1e-1
) %>%
  set_engine("xgboost", objective = "reg:squarederror")

## 4.2  Latin-hyper cube grid
set.seed(42)                               # reproducible Latin-hypercube
xgb_grid <- grid_latin_hypercube(
  trees(range = c(200L, 1200L)),
  tree_depth(range = c(3L, 10L)),
  min_n(range = c(2L, 40L)),
  loss_reduction(),
  sample_size = sample_prop(),
  finalize(mtry(), df_train),              # recipe applied internally
  learn_rate(range = c(-5, -1), trans = scales::log10_trans()),
  size = 400
)


# ──────────────────────────────────────────────────────────────────────
# 5.  Build workflow (recipe inside!)
# ──────────────────────────────────────────────────────────────────────

xgb_wf <- workflow() %>%
  add_recipe(brainage_recipe) %>%
  add_model(boost_mod)


# ──────────────────────────────────────────────────────────────────────
# 6.  Cross-validation split (5-fold × 2 repeats) and parallel backend
# ──────────────────────────────────────────────────────────────────────

set.seed(42)
train_cv <- vfold_cv(
  df_train,
  v       = 5,
  repeats = 2,
  strata  = interview_age
)

doParallel::registerDoParallel()


# ───────────────────────────────────────────────────────────────────────────
# 7. Hyper-parameter tuning - # this step runs the training and lasts for days 
# ───────────────────────────────────────────────────────────────────────────

set.seed(42)       # reproducible Latin-hypercube evaluation order
xgb_tuned <- tune_grid(
  xgb_wf,
  resamples = train_cv,
  grid      = xgb_grid,
  metrics   = metric_set(mae, rmse, rsq),
  control   = control_grid(verbose = TRUE, save_pred = TRUE)
)


# ───────────────────────────────────────────────────────────────────────────
# 8.  Post-tuning workflow and select best parameters
# ───────────────────────────────────────────────────────────────────────────

# quick leaderboard
show_best(xgb_tuned, metric = "mae", n = 10)

# select parsimonious params within one SE of best model
best_xgb_params <- xgb_tuned %>%
  select_by_one_std_err(metric = "mae", 
                        maximize = FALSE, 
                        tree_depth) 

save(best_xgb_params, file = "best_xgb_params.rds")


# ───────────────────────────────────────────────────────────────────────────
# 9.  Final fit and saving
# ───────────────────────────────────────────────────────────────────────────

# Finalize workflow and fit final model

final_xgb <- finalize_workflow(xgb_wf, best_xgb_params)
fit_workflow <- fit(final_xgb, df_train)

#(A) Workflow object – recipe + spec but **not** fitted weights
save(final_xgb, file = "final_xgb.rds")

# (B) Fully fitted model (weights + workflow) – ready for predict()
save(fit_workflow, file = "fit_workflow.rds")

# (C) Raw XGBoost booster – portable across languages (optional)
model_obj <- fit_workflow$fit$fit$fit
xgb.save(model_obj, "model_obj")
saveRDS(model_obj, file = "xgb_final_mod_RELEASE_5.1.rds")
