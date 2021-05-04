# STACI: A system to make black box models interpretable

Deep Learning models achieve state of the art results in machine learning tasks such as classification. However, such models are black box models, i.e., 
it is not possible to understand the logic behind their decision-making process. We propose STACI, a method that interprets the classification results of a black box model a posteriori. That is: given a dataset of points, and given a black box classification model, STACI can provide an explanation for the classification made by the model. Our idea is to emulate the complex classifier by surrogate decision trees. Each tree mimics the behavior of the complex classifier by overestimating one of the classes. 

![Alt text](./experiments/staci_figure.png "STACI explained")

This yields a global, interpretable approximation of the black box classifier. Our method provides interpretations that are at the same time general (applying to many data points), confident (generalizing well to other data points), faithful to the original model (making the same predictions), and simple (easy to understand). STACI stands for
**S**urrogate **T**rees for **A** posteriori **C**onfident **I**nterpretations.

## Usage

Download the code and import STACI:

```from staci import *```


To train the explainer, use the following code:

```explainer = STACISurrogates(max_depth=depth)```

```max_depth``` determines the maximum length of the interpretation (e.g. the depth of Decision Tree)

To fit the explainer, use:

```explainer.fit(x_train, y_train_black_box, black_box, features, target='target')```

The parameters are:
* the training data in the array-like form of the shape (number_of_samples, number_of_features) (```x_train```)
* the labels from the black box model in the array-like form of the shape (number_of_samples, ) (```y_train_black_box```)
* the black box model to be explained (must have  ```predict()``` method) ( ```black_box``` )
* the names of the features, one per dimension (```features```)
* the name of the feature that is the label of the model (```target```)


To explain an instance, use: 

```exp, ratio = explainer.verbose_predict(instance, predicted_label, features)```

Here, ```exp``` is the explanation, and ```ratio``` is the percentage of data points to which it applies.

## Reference

If you use or discuss our work, please cite

[N. Radulovic](https://github.com/nedRad88/), [A. Bifet](https://albertbifet.com/), [F. Suchanek](https://suchanek.name):
[“Confident Interpretations of Black Box Classifiers”](https://drive.google.com/file/d/1_3PhnwVLkBP4aZo4MXY6U15eceuUheXN/view?usp=sharing)
International Joint Conference on Neural Networks (IJCNN), 2021
