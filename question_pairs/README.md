![Quora Logo](https://storage.googleapis.com/kaggle-organizations/407/thumbnail.png?r=95)

# Musket example for Quora Question Pairs competition

This is Musket example for [Quora Question Pairs](https://www.kaggle.com/c/quora-question-pairs/overview)

Competition is aimed at predicting whether two questions asked on Quora are similar or not

## Launching from console

* Ensure you have Python 3.6, as well as Musket and Kaggle python packages with their dependencies installed, Keras installed and configured
* In experiment root folder - `quora-insincere-questions` - in command line type `musket fit -d`. `-d` tells Musket to download dependencies - Quora's insincere questiosn dataset in this case - before launching experiments. If you have more than one card available - don't forget to add `--num_gpus=N` parameter, where N is an amount of your available devices

## Launching from DS IDE

You can also edit and launch this example using our [DS IDE](https://musket-ml.github.io/webdocs/ide/getting_started/)

## Experiments

4 experiments are currently present

 * `pairs` - simple solution using LSTM architecture
 * `pairs_bert` - more advanced solution using Google BERT
 * `pairs_siamic` - solution using Siamic Network architecture over LSTM
 * `pairs_siamic_bert` - solution using Siamic Network architecture over Google BERT

## Modules and used capabilities

### datasets.py

This file contains dataset definions - methods decorated with `@datasets.dataset_provider(origin=...,kind=...)`. Such definitions can be generated by IDE based on a dataset
Besides that, it contains an example of the custom Dataset definition - `Questions2Outputs` сlass, which is used to provide inputs for siamic networks

### callbacks.py

This file contains `make_predictions()` method marked with `@after_fit` decoration. Given method should be called after training our model and which is aimed at making final prediction for data from file `test.csv`. You can use `@after_fit` decoration to mark any method in any module from `modules` folder, which should be called after finishing model training.
Present method does prediction for Bert

## Results

After fitting and prediction finished, please find file `data/predictions.csv` in project folder.
  
