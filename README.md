# mlrd project

[dvc setup](https://doc.dvc.org/start)

## cool tips

### `s3` compatibility

minio is s3 compatible. so during dvc setup,

```python
    dvc remote add -d storage s3://mybucket/dvcstore
```

works even if we don't change `s3` out

### dvc

- dvc tracks both data and model. neat!

## playing around with xgboost and linear regression

- using `data.csv`
- [xgboost guide](https://www.geeksforgeeks.org/machine-learning/implementation-of-xgboost-extreme-gradient-boosting/)
- [linear reg guide](https://www.datacamp.com/tutorial/sklearn-linear-regression)
