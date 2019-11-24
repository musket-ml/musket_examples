# Baseline for tgs-salt-identification-challenge

This project provides an example baseline for [tgs salt identification challenge](https://www.kaggle.com/c/tgs-salt-identification-challenge).

![Image](https://storage.googleapis.com/kaggle-media/competitions/TGS/drilling.jpg)

## How to run training

0) Ensure that Musket is installed, ensure that TensorFlow is installed, ensure that kaggle api is installed 
1) Checkout the project
2) Type `musket get_dependencies` in command line
3) Type `musket fit` in command line and wait while networks are being trained
4) Enjoy the results 

## How to make predictions

Now when you have trained network, you may want to do submission. To do this you need to type `musket predict --csv exp01 Test'


 
