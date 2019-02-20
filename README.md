This project contains all the code that is used to participate in the [HATEVAL](https://competitions.codalab.org/competitions/19935) task of [SEMEVAL-2019](http://alt.qcri.org/semeval2019/index.php?id=tasks). The datasets can be found at https://github.com/msang/hateval/tree/master/SemEval2019-Task5/datasets.

If you use any content from the following repo, please cite the following paper.


# REQUIREMENTS:

Python >= 3.5

[Glove](https://nlp.stanford.edu/projects/glove/)

[Fasttext](https://fasttext.cc/docs/en/english-vectors.html)

[Infersent](https://github.com/facebookresearch/InferSent)

[DeepecheMood++](https://codeload.github.com/marcoguerini/DepecheMood/zip/v2.0)

[Ekphrasis](https://github.com/cbaziotis/ekphrasis)

[text8](http://mattmahoney.net/dc/textdata)


# INSTRUCTIONS:

The arguments and the use can be found in run_model.py file.

### To run simple NaiveBayesModel:

```python run_model.py -clf='nb' -train='TrainFilePath' -test='TestFilePath' -sep='Seperator'```

### To run NaiveBayesModel using preprocessing:

Uncomment the stemmer lines in **preprocess** function in **run_model.py** and then run the following command:

```python run_model.py -pre='T' -clf='nb' -train='TrainFilePath' -test='TestFilePath' -sep='Seperator'```

### To run SVM using glove vectors:

Uncomment the 'import libglove.py' line in word_embeddings.py and then run the following command:

Comment the stemmer lines in **preprocess** function in **run_model.py** and then run the following command:

```python run_model.py -clf='svm' -emsrc='glove' -train='TrainFilePath' -test='TestFilePath' -sep='Seperator'```

### To run svm, xg-boost and logistic regression using infersent, run the following command.

Uncomment the 'import libinfersent.py' line in word_embeddings.py and then run the following command:

```python run_model.py -clf='all' -emsrc='infersent' -train='TrainFilePath' -test='TestFilePath' -sep='Seperator'```

### To run the models after preprocessing the text, use -pre='T'.

### To train on TrainFile and validate on TestFile, use -mode="dev".

### To just train on TrainFile and get cross validation results, use -mode='train'.

### To write the predictions to an output file, use -op='OutputFilePath'.
