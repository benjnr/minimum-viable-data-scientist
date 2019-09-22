# Model deployment

> Context:
>
> Ok my model is finally trained, time to deploy it.
>
> Let's draw the model lifecycle.

## Storing models

### Pickling it

Once the training is done, there is the difficult question of model storage. Most models are only available in python and not languages you would find in classic applications environments such as java or C++. The classic approach is then to serialize directly the model with **pickle**, which is the default python serialization format. A common best practice is to use **joblib** instead of the default library. The use is really simple. Here is a sample of code, which not only stores the model but also some performance metrics (pipelines and functions have been removed for clarity):

```python
import joblib
import json
from zipfile import ZipFile

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

random_state = 42
test_size = 0.33

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

rf = RandomForestRegressor()
rf.fit(X_train, y_train)

# compute the result
y_pred = rf.predict(X_test)
r2 = r2_score(y_test, y_pred)

# let's store some metadata
metadata = {}
metadata["dataset"] = "iris"
metadata["random_state"] = random_state
metadata["test_size"] = test_size
metadata["model_module"] = rf.__class__.__module__
metadata["model_type"] = lr.__class__.__name__
metadata["r2_score"] = r2

with open("metadata.json", "w") as f:
    json.dump(metadata, f)

with open("model.pkl", "wb") as f:
    joblib.dump(reg, f)

# here we want to store model, python env and metadata.
# so we take the three files and put them in a zip.
with ZipFile("model.zip", "w") as zip:
    zip.write("model.pkl")
    zip.write("metadata.json")
    zip.write("requirements.txt")
```

Let's write a simple inference script. Here lies one of the problem of pickel approach. When loading the model from the pickle file, the script should use the exact same package version you used in your training phase.

```python
import joblib
from sklearn.ensemble import RandomForestRegressor

if __name__ == "__main__":
    with open("model.pkl", "rb") as f:
        model = joblib.load(f)

    dummy_prediction = model.predict([[0, 0, 0, 0]])
    print(dummy_prediction)
```

The inference Dockerfile would be:

```dockerfile
FROM python:3.6

RUN mkdir /model
WORKDIR /model

COPY model.zip /model
COPY inference.py /model

RUN unzip model.zip
RUN pip install -r requirements.txt

CMD ["python", "inference.py"]

# $ docker run --rm mydocker
# [0.]
```

### ONNX

ONNX is the upcoming interchange format and should be used whenever possible. It can deal with neural network, but also [scikit-learn models and pipelines](<https://github.com/onnx/sklearn-onnx>) as well as xgboost. It allows the use of several inference runtime, most notably classic (from cpu/gpu) and Nvidia's TensorRT.

```python
import pandas as pd
import json

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from skl2onnx import convert_sklearn,
from skl2onnx.common.data_types import FloatTensorType

df = load_iris()
X, y = iris.data, iris.target

random_state = 42
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

rf = RandomForestRegressor()
rf.fit(X_train, y_train)

# compute the result
y_pred = rf.predict(X_test)
r2 = r2_score(y_test, y_pred)

# let's store some metadata
metadata = {}

metadata["dataset"] = "iris"
metadata["random_state"] = random_state
metadata["test_size"] = test_size

metadata["model_module"] = rf.__class__.__module__
metadata["model_type"] = lr.__class__.__name__

metadata["r2_score"] = r2

# Dataset is a 1*4 float matrix
initial_type = [('float_input', FloatTensorType([1, 4]))]
onx = convert_sklearn(
    model=rf,
    name="random forest",
    initial_types=initial_type,
    metadata=json.dumps(doc_string)
)

with open("rf_iris.onnx", "wb") as f:
    f.write(onx.SerializeToString())
```

ONNX transfer the algorithm into a computation directed acyclic graph.  An inference script would be:

```python
import onnxruntime as rt
import numpy

if __name__ == "__main__":
    sess = rt.InferenceSession("rf_iris.onnx")
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name

    value = numpy.array([0, 0, 0, 0], dtype=numpy.float32)

    pred_onx = sess.run([label_name], {input_name: value})[0]

    print(pred_onx)
```

What is interesting is that the python environment doesn't need to be imported when executing inference. The onnx inference library is enough. There are alternate runtimes, such as nvidia's TensorRT, and it can work on both CPU and GPU.

```dockerfile
FROM python:3.6

RUN mkdir /model
WORKDIR /model

COPY rf_iris.onnx /model
COPY inference_onnx.py /model

RUN pip install onnxruntime numpy

CMD ["python", "inference_onnx.py"]

# $ docker run --rm mydocker
# [0.]
```

## Model lifecycle

### Where does the model live

Once the model is trained, the next phase is obviously to go into production. There are three ways to integrate a machine learning model into an applications

- Classic API : The model answer to a client request
- Batch : The model predict  at regular intervals on a bulk of data
- Streaming : The model infer the result on a message it received

Obviously, there should be a discussion between product owners, developers and data scientists on how the ML feature should integrate into the general product. Batch is generally considered the easiest way to put a model into production. You will find a lot of way to put your model behind an API. However, it is rarely consumed by the product as is, and should be integrated within the data pipelines, usually a messaging or streaming system.

### ML is not the core of ML Engineering

Once the model environment is defined, there are a lot of additional tasks to be considered:

- reliability: the model should run, and run right
- scalability: the model should scale to the user base
- timely execution
- idempotency: it should be possible to time-travel within the model result
- performance tracking:
- feedback loop: on what conditions should a model be retrained
- monitoring: the "train" and "inference" phases logs should be accessible

![](https://developers.google.com/machine-learning/crash-course/images/MlSystem.svg)

(taken from *https://developers.google.com/machine-learning/crash-course/*)

ML engineers are tasked to put your work into  production. They are not often found since the job title is quite new. Try to identify who will take the responsibility to run your model and work on a common technical ground to begin with.

As a data scientist, you don't have the responsibility of running things into production. However, you still have the ownership of the model that you developed. As time passes, ML engineers should contact you to review the model performance metrics, and you should be able to monitor how

If you prepared well your code, with clear pipelines and container packaging, there should be little work left on the model training as a container can be run anywhere. The pressing matter is how to apply your model on new data, monitor it and trigger model retraining.

## How to infer

### API and streaming inference

API are straightforward to build. But building a great API takes real practice.

One way is to use already available tools such as [Seldon](https://github.com/SeldonIO/seldon-core), Berkerley's [Clipper](http://clipper.ai/) or Tensorflow dedicated [serving framework](https://www.tensorflow.org/tfx/guide/serving). Generic API frameworks such as [Flask](https://palletsprojects.com/p/flask/), [Connexion](https://github.com/zalando/connexion) or [FastAPI](https://github.com/tiangolo/fastapi) offers a greater control over the API. There are a lot of tutorial to get started on this path, but they don't usually cover security, monitoring and so on, so speak to developers and ML engineers so that they explain to you their requirements.

Once the model is deployed, you might run into performance issues. The usual answer is to adapt the inference servers with GPU/TPU, but that's Machine Learning engineers job.

### Batch inference and retraining

Batch inference is as easy as training. A luigi pipeline, packaged into a docker should do the trick most of the time. When batches are small enough, an API could also do the trick as a fallback solution.

Scaling batch inference is fairly straightforward too: Dask's predict or Spark UDFs are the way to go.

Retraining follows strictly the same logic: containers and luigi pipeline or API are usually enough.

### Wrapping up

Here is a matrix to help your choice:

| batch size \ Latency | near real time    | moderate    | slow                       |
| -------------------- | ----------------- | ----------- | -------------------------- |
| Streaming            | Web service + TPU | Web service | Web service                |
| mini batch           | NA                | Web Service | Task scheduler + framework |
| large batch          |                   |             | Task scheduler             |

## Todo list

1. I know how I will persist my model and why
2. I know the way the model will be integrated into the product
3. I embed my model in a pipeline+docker (batch) or an API depending on how data is served to the model

## Reading list

| type     | TL;DR                                           | date | link                                                         |
| -------- | ----------------------------------------------- | ---- | ------------------------------------------------------------ |
| reprod   | ML Logistics: a core book               | 2017 | [link](https://mapr.com/ebook/machine-learning-logistics/) |
