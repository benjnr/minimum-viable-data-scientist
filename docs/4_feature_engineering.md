# Feature engineering

> Context:
>
> Data is loaded, scripts are ready, palms are sweaty...
>
> And  your data is so huge it doesn't fit in and python crash.
>
> Before buying an expensive GPU that doesn't help at all, here are some pointers

## Feature engineering at scale

### Great datasets beats better models

Let's face it: heavy models don't beat feature engineering. A carefully crafted feature, that captures a specific aspect of the business life, is highly valuable. Apart from the obvious Pandas, there are some packages out there than can help to create new features, among which [featuretools](https://github.com/featuretools/featuretools/):

```python
import featuretools as ft

# creating and entity set 'es'
es = ft.EntitySet(id = 'sales')

# adding a dataframe
es.entity_from_dataframe(entity_id = 'bigmart', dataframe = combi, index = 'id')
```

Other great packages includes [tsfresh](https://github.com/blue-yonder/tsfresh) for time series and [featureforge](https://github.com/machinalis/featureforge). To address the balance of the dataset between several classes, there is a good package too [imbalanced-learn](https://imbalanced-learn.org/en/stable/index.html)

All of these packages allow building datasets that lasts and push simple models to the best result possible.

### The troubles of distributed computing

One of the core concern of pandas users is scalability ([link](<https://dev.pandas.io/pandas-blog/2019-pandas-user-survey.html>)). Wes McKinney, the author of the library, wrote a [great article](http://wesmckinney.com/blog/apache-arrow-pandas-internals/) about his vision on why pandas doesn't scale much.  Some pointers can still be found to optimise the run ([here](https://www.dataquest.io/blog/pandas-big-data/) and [there](http://www.unofficialgoogledatascience.com/2016/10/practical-advice-for-analysis-of-large.html)), but doing feature engineering on several gigs of data is still somewhat complex to apprehend. Fortunately, Pandas has an [ecosystem](<https://pandas.pydata.org/pandas-docs/stable/ecosystem.html#out-of-core>) that allow to scale through what is called "out-of-core" processing.

When loading a pandas dataframe, it gets stored in the RAM, basically the fast memory. However, this memory is usually up to 16-32 GB, which is not much sometime. Out-of-core processing uses instead the computer hard drive, which is usually around 1To. However, the read/write operation is also slower. That's why having a SSD drive is a good idea to boost this bottleneck. The main framework to do this kind of operation is [Dask](<https://docs.dask.org/>).

Another point of attention is how to optimize the computation. For example, a row-wise operation on the dataframe can be distributed easily. A shuffle operation (moving all rows) is one of the[ most expensive](<https://docs.dask.org/en/latest/dataframe-groupby.html>). So think about how you can compute your features.

The Dask team putted together some great [best practices](https://docs.dask.org/en/latest/dataframe-best-practices.html), a worthwhile read indeed.

### Dask Example

Dask is the most obvious choice when it comes to replacing pandas dataframes. It has a wide support from 200+ contributors and 5000+ commits, and can be used indifferently from small dataset to terrabytes on clusters.

```python
path_data = "NYC_taxi_2009-2016.parquet" # that's 35GB FYI !

@timeit
def load_df():
    df = dd.read_parquet([os.path.join(snappy_path, f) for f in os.listdir(snappy_path)])
    return df

@timeit
def describe(df):
    val = df.count()
    dd = val.compute()
    return dd

@timeit
def fare_to_euro(df):
    df["fare_amoun_euro"] = df.fare_amount*0.9
    df.compute()
    return df
```

(A decorator to time functions was taken from [here](<https://medium.com/pythonhive/python-decorator-to-measure-the-execution-time-of-methods-fa04cb6bb36d>))

```python
import dask.dataframe as dd

# ... here the functions
df = load_df()
# 'load_df'  510.72 ms
dd = count(df)
# 'describe'  468980.57 ms
```

Dask also offers a distributed version, which includes notable performance and task tracking

```python
from dask.distributed import Client
import dask.dataframe as dd

client = Client()

# ... here the functions
df = load_df()
# 'load_df'  463.89 ms
dd = count(df)
## 'describe'  269501.17 ms
```

| measure                              | time           |
| ------------------------------------ | -------------- |
| Simple Metadata computation          | 0:08:32.429791 |
| Simple dask column computation (add) | 0:10:34.628188 |
| Distributed metadata computation     | 0:04:35.804317 |
| Distributed dask column computation  | 0:06:18.56741  |

### Other tools

Several tools offers similar potential. Test them and see which poison please you.

- Vaex
- pandarralel
- modin
- Spark

## Todo List

1. If my computation takes time, think about how to optimise it
2. if my data is too large for the RAM, use dask
3. if my data is too large for my computer, use spark or dask distribued

## Reading List
