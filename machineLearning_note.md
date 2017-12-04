# Data Preprocessing

download data set from https://www.superdatascience.com/machine-learning/

## Missing data

`Lib: sklearn`

```python
from sklearn.preprocessing import Imputer
```

`Imputer`

```python
Definition : Imputer(missing_values="NaN", strategy="mean", axis=0, verbose=0, copy=True)
Type : Present in sklearn.preprocessing.imputation module
```

> Parameters ( __command + i__ to check help of class)

- missing_values

  The placeholder for the missing values. All occurrences of missing_values will be imputed. For missing values encoded as np.nan, use the string value “NaN”.

- strategy

  The imputation strategy.
  - If “mean”, then replace missing values using the mean along the axis. (平均)
  - If “median”, then replace missing values using the median along the axis.(中位数)
  - If “most_frequent”, then replace missing using the most frequent value along the axis.  (最高频数)

- axis

  The axis along which to impute.
  - If axis=0, then impute along columns.
  - If axis=1, then impute along rows.

- verboseThe imputation strategy.If “mean”, then rep

  Controls the verbosity of the imputer.

- copy

  If True, a copy of X will be created. 
  If False, imputation will be done in-place whenever possible. 
  Note that, in the following cases, a new copy will always be made, even if copy=False:
  - If X is not an array of floating values;
  - If X is sparse and missing_values=0;
  - If axis=0 and X is encoded as a CSR matrix;
  - If axis=1 and X is encoded as a CSC matrix.



## Encoding Categorical data

把可能性的value/String 转换为数字



 