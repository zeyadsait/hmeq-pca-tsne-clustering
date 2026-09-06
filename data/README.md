# Data

This folder is intentionally empty in the repository — the HMEQ dataset
is not redistributed here.

The scripts expect a file at:

```
data/HMEQ_Processed.csv
```

## How to get it

1. Download the original HMEQ ("Home Equity") dataset, e.g. from
   [Kaggle](https://www.kaggle.com/datasets/ajay1735/hmeq-data).
2. Clean / impute it the same way as in the Week 6 assignment this
   project builds on:
   - For every numeric column with missing values, create an imputed
     column (e.g. `IMP_MORTDUE`) filled with the median (or another
     reasonable strategy), plus a companion missing-value flag column
     (e.g. `M_MORTDUE`, 1 if the original value was missing, else 0).
   - One-hot encode the categorical columns `JOB` and `REASON` into
     `FLAG.Job.*` / `FLAG.Reason.*` dummy columns.
   - Keep `TARGET_BAD_FLAG` (binary default indicator) and
     `TARGET_LOSS_AMT` (loss amount, populated for defaulted loans) as
     the two target columns.
3. Save the result as `data/HMEQ_Processed.csv`.

If you already have a processed file from an earlier assignment, just
copy it here under that filename and the scripts will run as-is.
