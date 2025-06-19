# brain-age-prediction

This repo contains two scripts:

| Script | Description |
|--------|-------------|
| **training_validation.R** | Trains a model to predict brain age from imaging features |
| **predict_test_set.R**  | Loads a saved model and evaluates it or predicts on new data |

## Quick start

1. Train
Rscript train_brain_age.R \
        --train_csv data/train_features.csv \
        --n_trees 500 \
        --out_model models/brain_age.rds

# 2. Test / predict
Rscript test_brain_age.R \
        --model_path models/brain_age.rds \
        --test_csv data/test_features.csv