# Model management

> Context:
>
> Hey, I know how to train a model!
>
> You said experiment logs ? Scaling ? What's that ?

## Experimentation logging

### The need for a good experiment tracking

With all of the previous work, each code run should be reproducible with minimal overhead. Now, good software express itself through logs. It is pretty much the same with model development, also called experiment tracking. What should be tracked is all metrics useful to diagnose the model. Performance measures like Precision and Recall are obvious choice. Training time is also a good thing to log.

When the model is automatically updated with new data, experiment tracking allows to monitor it for bias, which is the expectation of any model.

### Example with MLFlow

The current open source tool of choice is [MLFlow](https://mlflow.org). It provides three separate toolkits: one for experiment tracking, one for project and the last one for deploying.

```python
import pyarrow
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import mlflow
from  mlflow.tracking import MlflowClient

if __name__ == "__main__":

    mlflow.set_tracking_uri("http://my.experiment.platform.com")

    client = MlflowClient()

    random_state = 42
    test_size = 0.33

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # persist the model
    with open("model.pkl", "wb") as f:
        joblib.dump(lr, f)

    # compute the result
    y_pred = lr.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    # now we can store some useful metadata: model type and class
    client.set_tag(run.info.run_uuid, "model_module", lr.__class__.__module__)
    client.set_tag(run.info.run_uuid, "model_type", lr.__class__.__name__)

    # log parameters and metric
    client.log_param(run.info.run_uuid, "random_state", random_state)
    client.log_param(run.info.run_uuid, "test_size", test_size)

    client.log_metric(run.info.run_uuid, "r2_score", r2)

    mlflow.sklearn.log_model(lr, "model")

    # finally, we can even store the model itself
    client.log_artifact(run.info.run_uuid, "model.pkl", "model")

    client.set_terminated(run.info.run_uuid)
```

Some alternatives exists like [ModelDB](https://mitdbg.github.io/modeldb/) by the MIT, or [Datmo](https://github.com/datmo/datmo).

## Machine learning at scale

When developing a model, several steps of the computation can be scaled:

- model training
- parameter search
- model prediction

### Scaling model training

The usual problem to scale model is either a lack of RAM or a lack of computing power. RAM limits has an obvious answer: out of core again. Another option is to fit the model with batches of data instead of full dataset in one go. Deep learning models are famous for that. In some case, it is possible to distribute training, like Decision Trees / boosted tree, Naive Bayes, Linear SVM... Sklearn list [some algorithms](https://scikit-learn.org/stable/modules/computing.html#incremental-learning) where incremental learning is available. This is done trough the use of a *partial_fit* method.

- regressions: Generalized Linear Models
- clustering: KMeans, spectral clustering, Logistic regression
- Boosted trees like XGBoost



```python
from dask.distributed import Client
from dask_ml.model_selection import IncrementalSearchCV
from sklearn.linear_model import SGDClassifier

from dask_ml.datasets import make_classification

client = Client()

X, y = make_classification(n_samples=5000000, n_features=20, chunks=100000, random_state=0)

model = SGDClassifier(tol=1e-3, penalty='elasticnet', random_state=0)

params = {'alpha': np.logspace(-2, 1, num=1000),
          'l1_ratio': np.linspace(0, 1, num=1000),
          'average': [True, False]}

search = IncrementalSearchCV(model, params, random_state=0)

search.fit(X, y, classes=[0, 1])

```

### Scaling parameter search

When tuning a model, the good strategy is to start with simple, brute force [scikit-learn optimisation](https://scikit-learn.org/stable/modules/grid_search.html).  If it takes too much time, there is two solutions:

1. adopt a faster parameter tuning
2. parallelize the parameter

Parallelization makes the computer train several models at the same time. Brute-force optimization like Grid Search or Randomized is basically a batch of independent training, so it does parallelize well. Some methods are a sequential execution of parallel computations, like [Evolutionary Algorithms](https://github.com/rsteca/sklearn-deap). And some are purely sequential and can't  be optimized, like simple [Bayesian optimization](https://github.com/hyperopt/hyperopt-sklearn), Gradient Descent or Particle Swarm Optimisation.

[dask-ml](https://ml.dask.org/hyper-parameter-search.html) provides optimized Grid Search and Random Search that are really easy to implement. However, they only work if the model fits in memory.

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

pipeline = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier())])

grid = {'vect__ngram_range': [(1, 1)],
        'tfidf__norm': ['l1', 'l2'],
        'clf__alpha': [1e-3, 1e-4, 1e-5]}

```

Here is an example of pure dask parallel optimization:

```python
#reading the csv files
import dask.dataframe as dd
from dask_ml.linear_model import LinearRegression
from dask.distributed import Client
import dask_ml.joblib
import dask_searchcv as dcv
from sklearn.externals.joblib import parallel_backend

df = dd.read_csv('blackfriday_train.csv')
test=dd.read_csv("blackfriday_test.csv")

#defining the data and target
categorical_variables = df[['Gender', 'Age', 'Occupation', 'City_Category', 'Stay_In_Current_City_Years', 'Marital_Status']]
target = df['Purchase']

#creating dummies for the categorical variables
data = dd.get_dummies(categorical_variables.categorize()).compute()

#converting dataframe to array
datanew=data.values

#fit the model
lr = LinearRegression()
lr.fit(datanew, target)

#preparing the test data
test_categorical = test[['Gender', 'Age', 'Occupation', 'City_Category', 'Stay_In_Current_City_Years', 'Marital_Status']]
test_dummy = dd.get_dummies(test_categorical.categorize()).compute()
testnew = test_dummy.values

#predict on test and upload
pred=lr.predict(testnew)

client = Client() # start a local Dask client

with parallel_backend('dask'):

    # Create the parameter grid based on the results of random search
     param_grid = {
        'bootstrap': [True],
        'max_depth': [8, 9],
        'max_features': [2, 3],
        'min_samples_leaf': [4, 5],
        'min_samples_split': [8, 10],
        'n_estimators': [100, 200]
    }

    # Create a based model
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor()

# Instantiate the grid search model
grid_search = dcv.GridSearchCV(estimator = rf, param_grid = param_grid, cv = 3)
grid_search.fit(data, target)
grid_search.best_params_
```

## Testing a model

Now the model is trained and even optimized, which is great. However, models have biases. Several tools allow to peek into the models, but alow

| Package name                                        | aimed at                            | task                              |
| --------------------------------------------------- | ----------------------------------- | --------------------------------- |
| [netron](https://github.com/lutzroeder/netron)      | deep learning                       | model diagnosis                   |
| [PyCM](https://github.com/sepandhaghighi/pycm)      | classification prediction           | a lot of confusions matrices      |
| [Eli5](<https://eli5.readthedocs.io/en/latest/>)    | scikit, XGBoost, LightGBM, Keras... | One of the first package to shoot |
| [SHAP](<https://github.com/slundberg/shap>)         | idem                                | idem                              |
| [XAI](<https://ethicalml.github.io/xai/index.html>) | A newcomer                          |                                   |

## Todo list

1. My experiments are logged
2. I analyse if my model training will need to be scaled, and if so, I know the tools of the trade
3. I test my model and prediction, and write a report about it's fairness

## Ressources

https://tomaugspurger.github.io/scalable-ml-02.html