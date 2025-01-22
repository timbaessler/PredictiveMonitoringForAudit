## Preprocessing

```python
from src.preprocessing.utils import read_xes, get_time_attributes, get_seq_length

df = read_xes('<LOG PATH>')

# Feature Engineering
df = get_time_attributes(df)
df = get_seq_length(df)
```

## Modeling

```python
from src.models.encoding import Aggregation
from src.models.bucketing import TimeBucketing
from sklearn.model_selection import train_test_split
from src.models.cv_models import CrossValidation

# Time Bucketing
bdays = 7 # predict 7 business days before threshold
bucketer = TimeBucketing(offset=bdays, deadline_col='deadline')
df = bucketer.fit_transform(df)

# Aggregation
static_cat_cols = list(['case:young farmer','case:penalty_AJLP','case:small farmer'])
dynamic_cat_cols = list(['org:resource', 'concept:name', 'success', 'doctype', 'subprocess'])
num_cols = list(['case:penalty_amount0', 'month', 'weekday', 'hour', 'time_since_first_event'])
agg_transformer = Aggregation(num_cols, static_cat_cols, dynamic_cat_cols)
X, y = agg_transformer.fit_transform(df)
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Fit Classifier
classifier = 'XGBoost'
crossval = CrossValidation(classifier=classifier, param_dict=param_dict[classifier], cvs=5)
clf = crossval.get_classifier()
clf.fit(X_train, y_train)
# Predict
y_pred = clf.predict(X_test)
```
