[![DOI](https://zenodo.org/badge/485132738.svg)](https://zenodo.org/badge/latestdoi/485132738)

# Overview
The code and data in this repository is for the Perspectives Class Project, Spring 2022, at the University of Chicago. 

The purpose of this class project is to evaluate the Fringe Effect of Social Movement Organizations (SMOs) in Online Social Movements. More specifically, I want to study the BlackLivesMatter movement, and see how SMOs' popularies and display of emotion changed after the death of George Floyd.

The [code](https://github.com/LuZhang0128/Perspectives/blob/main/analysis.ipynb) is written in Python 3.7 on Google Colab. <br>
All data used in this project, as well as the saved outputs, can be found [in this folder](https://drive.google.com/drive/folders/1Y1267sa7shpWpW31RBivD_CmDNQ07AeC?usp=sharing). 

# Dependencies
All the dependency (required packages) are in the `Import Packages` section in Google Colab code, and is also listed below:

```
!pip install -U git+http://github.com/UChicago-Computational-Content-Analysis/lucem_illud.git --quiet
!pip install NRCLex --quiet
!pip install -U kaleido --quiet
!pip install transformers --quiet
!pip install imbalanced-learn --quiet
import imblearn
import pandas as pd
import re
import nltk
nltk.download('punkt')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import wordcloud
import spacy
import gensim
import numpy as np 
import seaborn as sns
import sklearn.metrics.pairwise 
import sklearn.manifold 
import sklearn.decomposition 
import lucem_illud
import datetime as dt
import pandas as pd
import lucem_illud
import math
import sklearn.feature_extraction.text
import sklearn.feature_extraction
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve
import matplotlib.pyplot as plt
import gensim
import wordcloud
from lucem_illud.processing import normalizeTokens, trainTestSplit, word_tokenize, sent_tokenize
from nrclex import NRCLex
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
from scipy.special import softmax
import csv
import urllib.request

%matplotlib inline
```

# Load and Clean Data
I've collaborated with my friend and have written a scraper for Twitter data. I am currently unable to share the scraper with the class.
<br>

This scraper only randomly send back a certain amount of Tweets per day every time I try to make request through the code. For this class project, I just used a sample of data (one output file by running the scraper). The raw sampled data can be found [here](https://drive.google.com/file/d/19umvDYuu1o6uIr3xz1m0qwLay4qgeQ5r/view?usp=sharing). This project can expand the scale by combining with multiple output files in the data folder. 
<br>

You can pre-process the data by running the `Load and Clean Data` section in Google Colab code. 
<br>

In this section, I removed urls and uninformative information like websites and '@'s. I then splited data into before and after the death of George Floyd. I then tokenized and normalized all tweets using the packages developed by Prof. James Evans' research team. 
<br>

The pre-processed data can be found [here in csv](https://drive.google.com/file/d/1Szztx7LW-QGuvejjWfl9eea9oJ_whtFQ/view?usp=sharing) or [here in pkl](https://drive.google.com/file/d/1bhXF7WidraR3qkpzv9giXaR5WZGov78k/view?usp=sharing). 

# Basic Data Description
In this sampled dataset, there are in total 123517 observations. Let's get a quick look at the distribution of the data, as well as the wordclouds. The distribution can be reproduced by running the `Basic Plots` section in Google Colab code, and the wordclouds can be reproduced by running the `Word Cloud` section in Google Colab code. Note that this basic description only get people to know the dataset better, and is not related to the final answer to the research question.
<br>

### Distribution
The distribution of number of tweets per day is as below: <br>
<img src="https://github.com/LuZhang0128/Perspectives/blob/main/figs/number_of_posts_per_day.png" width=60% height=60%>
<br>

This plot, however, does not necessarily reflect the true distribution of posts. After 2016, the number of posts is more likely to be limited by the algorithm instead of telling us the true trend. If we want to study the trend, [Google Trend](https://trends.google.com/trends/explore?date=2007-12-31%202022-04-24&geo=US&q=blacklivesmatter) is an alternative source.
<br>

### WordClouds
The wordcloud generated using all tweets in the sampled dataset as below: <br>
<img src="https://github.com/LuZhang0128/Perspectives/blob/main/figs/wordcloud_all.png" width=40% height=40%> <br>
The wordcloud generated using tweets before (right) and after (left)the death of George Floyd (2020-05-25) are as below: <br>
<img src="https://github.com/LuZhang0128/Perspectives/blob/main/figs/wordcloud_before.png" width=40% height=40%> 
<img src="https://github.com/LuZhang0128/Perspectives/blob/main/figs/wordcloud_after.png" width=40% height=40%> 
<br>

It is interesting to see words such as "wildhorse," "wild," "horse," and "land" in the first word cloud, and not in the second one. This shift indicates a slight change in topic before and after the identified event (death of George Floyd). It also shows that I'm on the right track.


# Machine Learning Model for Classification
I trained four supervised machine learning models to classify all Twitter accounts to four categorties: 
1) Social Movement Organization (SMO)
2) Other Organization
3) Social Movement Activists
4) Other Individuals

Three individual research assistants and I are currently labelling the training dataset. The manually labelled data (in progress) can be found in [this spreadsheet](https://docs.google.com/spreadsheets/d/1Re-t0Tc7OLDYzt5cJWZcztvwllyMw3mzOWhXoOYnnQM/edit?usp=sharing). One Twitter account is at least coded by two people. Any discrepancies are checked and resolved by a thrid people. <br>

The distribution of the manually labeled data is plotted in R-studio using [this code](https://github.com/LuZhang0128/Perspectives/blob/main/label_distribution.Rmd) and is as follows: <br>
<img src="https://github.com/LuZhang0128/Perspectives/blob/main/figs/labeled_distribution.png" width=40% height=40%> 
<br>

There are 55 (12%) Social Movement Organizations, 67 (15%) Other Organizations, 84 (19%) Social Movement Activists, and 244 (54%) Other Individuals. Most accounts are everyday people. These 450 accounts are responsible for 3,566 (2.89%) tweets among all 123,517 tweets. <br>

The trainings of the four machine learning model (Random Forest Classifier, Logistic Regression, Support Vector Classification (SVC), and Multinomial Naïve Bayes algorithm) are in the Google Colab code, under the `Machine Learning Model` section. <br>

The Confusion Matrices are of the models are: <br>
<img src="https://github.com/LuZhang0128/Perspectives/blob/main/figs/confusion_matrices%20.png" width=40% height=40%> <br>
and the accuracy scores are: <br>
<img src="https://github.com/LuZhang0128/Perspectives/blob/main/figs/accuracy_scores%20.png" width=40% height=40%> <br>

The confusion matrices show that all four models experience high errors when differentiating SMOs’ accounts from Individual Activists’ accounts. When the true label is 3 (Social Movement Activists), the models are more likely to falsely classify it as SMO than the other two categories. Meanwhile, the models can tell Individual Activists apart from everyday people. When the true label is 4 (Other Individual), the model is doing a good job doing the classification. The error rates of classifying them into other three categories are similar, meaning that the model did not find significantly higher seminaries between everyday people and individual activists. <br>
Based on the accuracy scores, the Random Forest classifier achieved the highest accuracy score. However, all four models have better performance on the training sets compared to the testing sets. This suggests over-fitting of the models. The accuracy scores on the testing sets are around 70%. <br>

# Emotion Classification
Since I haven't finished the classification model, I can only examine the emotion display of all Twitter accounts. Like the word clouds, I look at emotions as a whole, and emotions before and after the event. I used the NRCLex package, which is a dictionary-based emotional classification algorithm. Later, I will also consider using a neural-network-based algorithm to achieve higher classification accuracy. The emotion classifications can be reproduced by running the `Emotion Classification Before and After the Event` section in Google Colab code. 
<br>

The emotion classification generated using all Twitter accounts in the sampled dataset as below: <br>
<img src="https://github.com/LuZhang0128/Perspectives/blob/main/figs/emotion_all_count.png" width=60% height=60%> <br>
The wordcloud generated using all Twitter accounts before (up) and after (down)the death of George Floyd (2020-05-25) are as below: <br>
<img src="https://github.com/LuZhang0128/Perspectives/blob/main/figs/emtion_before_percentage.png" width=60% height=60%> 
<img src="https://github.com/LuZhang0128/Perspectives/blob/main/figs/emtion_after_percentage.png" width=60% height=60%> 
<br>

From the plots above, we can see that both the percentage of `fear` words and `anger` words decrease after the death of George Floyd. This finding is counterintuitive and is different from Bail's conclusion about Fringe Effect, that organizations tend to display fearful and angry words to attract public's attention). After further splitting the accounts, I want to see if this emotion classification results will be different for Social Movement Organization (SMO), Other Organization, Social Movement Activists, and Other Individuals.

# Cite this Repository
You can cite this repository via the doi on the very top of the README file.

