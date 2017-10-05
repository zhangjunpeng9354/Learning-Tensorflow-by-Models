# Import Data to TensorFlow

This is a tutorial on importing data to TensorFlow.

In my understanding, importing data to TensorFlow is one of the most
important steps when using TensorFlow and Deep Learning models for
engineering. Most current models have already been implemented, even
the weights are available. Utilizing these models is about either
re-training the models or fine-tuning the models on new datasets.
Importing data to TensorFlow is the step prior to starting the explorations.

## Introduction

Generally, there are four approaches to import data to TensorFlow Program:

1. using the Datasets API,
2. feeding,
3. reading from files,
4. preloaded data.

Some of these approach are more suitable for small datasets, while others may
be designed for large machine learning problems with distributed files.

## Using the Datasets API

Input pipeline is a mechanism to build a sequence of elements, for example,
in image recognition problems, each element is a bundle of a training sample
and its corresponding label. An input pipeline includes a
```tf.contrib.data.Dataset``` for representing the elements, and two mechanisms
for operating the dataset. These two mechanisms are utilized for transformation
and iteration over the dataset respectively.





## Related Guides

Official documents provide some guides on this topic in the following links:

1. [Importing Data](https://www.tensorflow.org/programmers_guide/datasets),
2. [Reading data](https://www.tensorflow.org/api_guides/python/reading_data)