# SDG Auto Labeller

## Python application to automate matching of Sustainable Development Goals to academic papers

SDG_auto_labeller.py is a Python module that can be run from the command line to automatatically match Sustainable Development Goals to academic papers about international development interventions. It outputs a CSV file with predicted labels and probabilities.

The model uses NLP and supervised learning techniques to create the labels, automating a time-consuming process that is currently done by human labellers.

## Project overview 

### Developing the machine learning model

Descriptions and titles of each of the 17 Sustainable Development Goals were tokenized into n-grams (1 - 3) and transformed into an l2-normalized TF-IDF vector. 

Basic descriptions and titles of over 3,000 were also tokenized and transformed into TF-IDF vectors using the same vocab / n-grams from the SDG tokenization process.

A feature matrix was created containing the following features:
- Cosine similarity matrix of each review paper with each SDG
- SDG labels with the highest TF-IDF values corresponding to the same keyword in the academic paper (e.g. if "Aid" had the highest TF-IDF score in review paper 1, the feature extraction function would find the SDG where "Aid" has the highest TF-IDF value). 

The feature matrix was divided into a training and holdout set.

An XGBoost model was trained on the training set using 5-fold cross validation and hyperparameters were tuned using hyperopt (a Bayesian optimisation library).

The best parameters were then used to train another XGBoost model on the entire dataset and scored on a the holdout set. The accuracy score was 81%. 

### Fine tuning the model to improve precision and creating the application

Given that the ultimate purpose of the project was to reduce the time spent by human labellers, we included an extra step in adjusting the thresholds to prioritise precision instead of accuracy. 

The idea was that 81% accuracy isn't accurate enough to replace or assist human labellers. However, a more practical approach would be to achieve 95%+ precision by adjusting probability thresholds and only label those papers the model was most confident about. This would signficantly reduce the human labellers workload by allowing them to focus only on a subset of papers.

Finally, the XGBoost model was pickled and a Python module was created to automate the process of labelling new data (in Stata format) directly from the command line using the saved model.

## Getting Started / installation

The project consists of the following files:

- **SDG_auto_labeller.py** - the Python module containing the full code to be run from the command line.
- **xgb_sdg_model.pickle** - the pre-trained and pickled XGBoost model
- **SDG_list.pickle** - a pickled list of SDGs and their descriptions
- **SDG_goal_heading.pickle** - a pickled list of SDG headings
- **SDG_label_encoder.pickle** - the trained label encoder for use with the existing trained model
- **SSAevidenceNov_181019.dta** - Stata file containing the raw data from the systematic review papers
- **SDGresults_py.xlsx** - Excel file containing the final labels

To install and run the module using the pre-trained model, simply download the first 6 modules to the same folder and run the auto labeller from the command line after navigating to the folder, e.g.:

```
$ python SDG_auto_labeller.py
```

## Prerequisites

To run the code, there are a number of dependencies. A Python 3 Anaconda distribution should cover most of the required libraries.

You may need to pip install the following:

```
NLTK
Pandas
NumPy
Matplotlib
Hyperopt
Seaborn
Pickle
SciPy
XGBoost
Scikit-learn
```

## Acknowledgments

This project was completed in collaboration with Richard Appell, Research Associate at 3ie - the impact evaluation charity.
