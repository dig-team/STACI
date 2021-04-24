# STACI

**S**urrogate **T**rees for **A** posteriori **C**onfident **I**nterpretations.

STACI interpretation of Black Box classifiers using multiple surrogate Decision Trees.

![Alt text](./experiments/staci_figure.png "STACI explained")


## Usage

Download the code and import STACI:

```from staci import *```


To train the explainer, use the following code:

```explainer = STACISurrogates(max_depth=depth)```

```max_depth``` determines the maximum length of the interpretation (e.g. the depth of Decision Tree)

To fit the explainer, provide: training data (```x_train```), labeled using black box model (```y_train_black_box```), black box model to be explained ( ```black_box``` ),
and feature names (```features```), specifically the label column name (```target```)
```explainer.fit(x_train, y_train_black_box, black_box, features=attrs, target='target')```


To explain the instance: 

```exp, ratio = explainer.verbose_predict(instance, predicted_label, features)```


## Reference

[N. Radulovic, A. Bifet, F. Suchanek, "Confident Interpretations of Black Box Classifiers",
International Joint Conference on Neural Networks (IJCNN),2021 ](https://drive.google.com/file/d/195bgAx_uFGCqQakkjLYsPIa0jeOk5Tfy/view?usp=sharing)