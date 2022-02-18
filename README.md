# EVI

Expression and Velocity Integration

## Introduction
<p align="center">
  <img src="/doc/overview.png"/>
</p>

```
model = evi.tl.EVI(adata = adata, x1key = x1key, x2key = x2key, X1 = X1, X2 = X2, int_method = int_method,
            int_method_params = int_method_params, eval_method = eval_method, eval_method_params = eval_method_params, logX1 = logX1,
            logX2 = logX2, labels_key = labels_key, labels = labels, n_jobs = n_jobs)

W, embed = model.integrate()
df = model.evaluate_integrate()
```
