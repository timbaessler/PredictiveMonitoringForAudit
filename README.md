# Predictive Process Monitoring For Internal Audit

This repository contains supplementary material for the article [Predictive Process Monitoring for Internal Audit: Forecasting Payment Punctuality from the Perspective of the Three Lines Model](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4080238).

## Abstract
Research in predictive process monitoring exists mainly from the perspective of process and risk owners. Existing guidelines from the Institute of Internal Auditors (IIA) omit process mining or process predictions for internal audit. Attempting to bridge this gap, we implement a framework which redefines the role of internal audit using predictive process monitoring within the Three Lines Model. The framework takes into account that the internal audit function should not interfere with the obligations of the first two lines, but still reduces risk and prevents undesired outcomes. Since many outcomes of interest in internal audit (like payment punctuality, payment discounts, receipt of goods, deliveries etc.) have fixed deadlines, we use fixed time buckets for training and runtime implementation. This bucketing approach moves the prediction towards the outcome and leaves audit clients enough time to prevent the undesired outcome. Using machine learning methods and implementing a classical internal audit use case, payment punctuality, with a publicly available event log, we show that internal audits can use process predictions to provide assurance, reduce risk and prevent undesired outcomes.

## Keywords: 
Internal Audit, Machine Learning, Predictive Auditing, Process Mining, Corporate Governance Structure, Three Lines Model

## Workflow
We propose to use remaining-time bucketing of the traces. We expect that the point of time of the outcome is exactly known (e.g., term of payment, date of delivery) (see Figure 2). Here one would predict, if the case reaches a time-threshold with only a certain amount of time left. During runtime, an ongoing partial trace is used for predictions if it matches the chosen remaining time bucket (e.g., 7 days). If the trace is predicted to have a positive label (i.e., an undesired outcome) then the respective department will get notified (via for example E-mail). This approach does not require judgement of the auditor, and therefore an automated message via E-Mail or directly in the ERP system could be implemented.
![workflow](https://github.com/timbaessler/PredictiveMonitoringForAudit/assets/94218704/7e202889-1891-4957-be41-236ee56decfa)

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
import xgboost

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

# Fit Classifier
clf = xgboost.XGBClassifier()
clf.fit(X, y)
```
