# NLP Sequence To Sequence - Machine translation example

## Important! This example uses the latest incubation musket_core and musket_text libraries. To get them:
 * Clone repository https://github.com/musket-ml/musket_core/
 * Execute `pip install .` in it's root
 * Clone repository https://github.com/musket-ml/musket_text
 * Execute `pip install .` in it's root
 
This example is aimed at Machine Translation neural network demonstration

Explanation and some non-Musket examples can be found [here](https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html) or [here](https://towardsdatascience.com/understanding-encoder-decoder-sequence-to-sequence-model-679e04af4346)

This example model has:
 * Tokens-to-Indexed and Google embeddings as input
 * 2 cudNN LSTM layer
 * One convolutional layer with kernel size 3 and softmax activation to get resulting word index
 
## Modules (modules/) folder in project
 * custom_preprocessors.py - contains one preprocessor, which expand dimensions of the LSTM output, since convolution doesn't work with data not having channels
 * datasets.py - contains dataset provider with input file (English > Russian sentence set by default) parsing logics
 * example.py - example of translating a string containing a single sentece. It demonstrates loading of the model and the vocabulary and using them to predict and build a resulting sentence
 * translate.py - demonstrates a callback, which would be called after finishing the network fitting process. Translates and prints the small subset of the original dataset

 
 