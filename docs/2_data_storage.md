# Data storage and access

> Context :
>
> Your project is booted correctly, yay!
>
> You even managed to see how to accept contributions from your colleagues !
>
> But now you got a HUGE trove of data to store, sitting in an external drive.
>
> How to properly manage that ?

## Data format

### Common storage formats are not enough

The most common storage formats are **CSV** and **JSON**. They are text based, which means that you can read it when opening the file with a text editor. CSV is a tabular format, whereas JSON stores objects.

Both formats are prone to errors. For example, CSV usually separates columns with comma. However, semicolon or tabs are also used since comma can be used for decimal numbering. Another example is that Windows and Linux have different line endings in text files (is anyone struggling with *\r\n* for Windows and *\n* for Linux ?).

CSV and Json doesn't have features for optimised manipulation and storage. They don't compress data, neither do they have features like schema storage, query optimisation, splittable query and so on. This gives a significant overhead when processing data.

### Storage formats

With the advent of distributed computations, a lot of work has been made to improve data storage. It has added data compression as well as computation optimisation. Let's talk a bit about them.

**Apache Parquet** is a tabular format. It replaces efficiently CSV. In [this benchmark](https://blog.openbridge.com/how-to-be-a-hero-with-powerful-parquet-google-and-amazon-f2ae0f35ee04), it gives a hefty 34x speedup in data query and 87% storage space savings. It is already integrated within pandas; are quite easy to use too. For example, a pandas dataframe can be stored in parquet quite easily:

```python
import pandas as pd
from sklearn import datasets
import fastparquet

iris_dataset = datasets.load_iris()

df = pd.DataFrame(
    iris_dataset.data,
    columns=iris_dataset.feature_names
)

df.to_parquet('df.gzip.parquet', compression='gzip')
```

**Apache Avro** is another highly efficient format. It stores basic object, quite similar to json. It [also gives](https://fr.slideshare.net/oom65/file-format-benchmarks-avro-json-orc-parquet) a boost to queries and storage space saved. However, it also needs a schema to be defined, as shown in the sample code below (taken from [fastavro library](https://github.com/fastavro/fastavro))

```python
from fastavro import writer, reader, parse_schema

schema = {
    'doc': 'A weather reading.',
    'name': 'Weather',
    'namespace': 'test',
    'type': 'record',
    'fields': [
        {'name': 'station', 'type': 'string'},
        {'name': 'time', 'type': 'long'},
        {'name': 'temp', 'type': 'int'},
    ],
}
parsed_schema = parse_schema(schema)

# 'records' can be an iterable (including generator)
records = [
    {u'station': u'011990-99999', u'temp': 0, u'time': 1433269388},
    {u'station': u'011990-99999', u'temp': 22, u'time': 1433270389},
    {u'station': u'011990-99999', u'temp': -11, u'time': 1433273379},
    {u'station': u'012650-99999', u'temp': 111, u'time': 1433275478},
]

# Writing
with open('weather.avro', 'wb') as out:
    writer(out, parsed_schema, records)

# Reading
with open('weather.avro', 'rb') as fo:
    for record in reader(fo):
        print(record)
```

These formats are straightforward and boost a lot the confidence of data storage, so use it without reserve.

## Storage spaces

### Where is my data?

There are no systems that fits the requirements from all data usage within a company. That's why datalakes are usually divided between several zones. This design is the work of data architects, helped by data engineers. Data scientists have usually access to a specific space:

| Zone              | use case                                         | Example                                                      |
| ----------------- | ------------------------------------------------ | ------------------------------------------------------------ |
| Landing / Staging | Temporary storage for data ingestion             | - CSV/JSON ingested with nifi<br />- stored in HDFS          |
| Raw               | Permanent storage                                | - HDFS + parquet                                             |
| Curated           | Data trusted for business use<br />enriched data | - quality processing pipelines with airflow/spark<br />- HDFS + Parquet |
| Analytics/Sandbox | Available to data scientists                     | - dataset builded with airflow/spark<br />S3 +Parquet        |
| Production        | Serve data to many apps                          | - database builed with airflow/spark <br />- Stored in distributed database <br />- HBase, Cassandra, Kudu... |

Within this analytics zone, data scientists should have the free reigns to experiment. There should be personal, projects and public spaces. The public space would include:

- all company's relevant data, obviously
- open data
- Machine learning benchmark datasets (useful for transfer learning)
- datasets published by other data scientists

Obviously, this storage space would also allow to store all other artifacts generated by a data science project, such as a models.

### Storing in HDFS and S3

Now let's see how the analytics space integrates to your data science workflow. There is two main storage technologies out there, hdfs and S3.

HDFS is the basic datalake filesystem. It offers a robust storage space within an Hadoop platform, allowing distributed data computation with Hive and Spark. Once the [setup is done](https://wesmckinney.com/blog/python-hdfs-interfaces/), it is fairly easy to store or retrieve files within it with a python hdfs client (from [Apache Arrow](https://arrow.apache.org/docs/python/filesystems.html#hadoop-file-system-hdfs)):

```python
import pyarrow as pa

fs = pa.hdfs.connect(host, port, user=user, kerb_ticket=ticket_cache_path)

# read small dataset
with fs.open("/data/database_dump.parquet", 'rb') as f:
    data = f.read()

# write in local dir
with open("data/raw/database_dump.parquet", 'wb') as f:
    f.write(data)
```

Amazon S3 means Simple Storage Service. It is a simpler file system, accessed through HTTP protocol. It's file organization is quite simple: buckets are drop-in storage space. Each file is stored and reference using a key. There is no sub-folder, however, a key can be *my_path/my_file.parquet*. The official package is [boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html). With the [setup done](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/quickstart.html#configuration), it gives that:

```python
import boto3

s3 = boto3.client('s3')

with open("data/raw/database_dump.parquet", 'wb') as f:
    s3.download_fileobj('my_bucket', 'database_dump.parquet', f)

```

There are several s3 compatible storage systems, for example [Minio](https://min.io/) allows to build an efficient s3 server through docker deployments.

### Managing a distant storage

The previous chapter presented how to organize a local repository. How does that work with a distant repository ? What needs to be stored here is everything that is not stored in git - that is to say the data and target folders. Now the question is: when to store all of that? There are three main timing when

- **experiments** are when a data scientist run a training and when to save the result. This is a manual process. A good practice is to automate the naming and storage within the code (see next chapter)
- **snapshots** are saved automated by tools like Jenkins or Airflow. Each time a data scientist merge it's code to designated branches (master and development usually), a *build* is triggered and the results should be stored

Snapshots is a really powerful idea that's helps productions. A common pattern would be that each time a snapshot is taken, the eventual models are then deployed as an API.

```shell
# manual storage
s3://my_project/experiments
├── manual_experiment_20190101_01 # name automatically generated by the code
|   ├── data
|   │   ├── external
|   │   ├── interim
|   │   ├── processed
|   │   └── raw
|   └── target
|       ├── logs
|       ├── models
|       └── graphs
|
└── manual_experiment_20190101_02

# from CI/CD pipeline on dev branch on each merge
s3://my_project/snapshots_development
├── <git_hash>
|   ├── data
|   │   ├── external
|   │   ├── interim
|   │   ├── processed
|   │   └── raw
|   └── target
|       ├── models
|       └── graphs
|
└── <another_git_hash>

# from CI/CD pipeline on master branch on each merge/tag
s3://my_project/snapshots_master
├── <git_hash/tag>
|   ├── data
|   │   ├── external
|   │   ├── interim
|   │   ├── processed
|   │   └── raw
|   └── target
|       ├── models
|       └── graphs
|
└── <another_git_hash/tag>

```

In this chapter, it was suggested to handle data control with the code. There are several toolkits that can help to do this task as well. The most interesting is [data version control](<https://dvc.org/>) which offers a git like client. There is also [pachyderm](<https://pachyderm.io/>), which suppose some choices about architecture. However, both of these means additional bash scripting. It is more complex to handle and harder to track than simple python code files, hence this book choice.

## Todo list

1. Find where you have to store your artifacts
2. Choose optimized format - Avro or Parquet.
3. Organize the folders
4. Automate the experiments/snapshot saves
5. If peoples keep data dumps on their computer, you screwed

## Reading list

| type        | TL;DR                                            | year | link                                                         |
| ----------- | ------------------------------------------------ | ---- | ------------------------------------------------------------ |
| File format | Overview of Hadoop format                        | 2015 | [link](https://databaseline.bitbucket.io/an-overview-of-file-and-serialization-formats-in-hadoop/) |
| File format | benchmark: parquet vs csv                        | 2017 | [link](https://blog.openbridge.com/how-to-be-a-hero-with-powerful-parquet-google-and-amazon-f2ae0f35ee04) |
| File format | benchmark: json vs orc, avro, parquet            | 2016 | [link](https://fr.slideshare.net/HadoopSummit/file-format-benchmark-avro-json-orc-parquet) |
| File format | benchmark: text vs orc avro parquet              | 2016 | [link](http://www.svds.com/dataformats/)                     |
| File format | benchmark: kudu, hbase, parquet, avro            | 2017 | [link](http://db-blog.web.cern.ch/blog/zbigniew-baranowski/2017-01-performance-comparison-different-file-formats-and-storage-engines) |
| data mngmt  | google for datasets (great data catalog example) |      | [link](https://toolbox.google.com/datasetsearch)             |
| data mngmt  | list of ML research dataset by google            |      | [link](https://ai.google/tools/datasets)                     |
