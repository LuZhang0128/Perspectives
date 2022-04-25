# Overview
The code and data in this repository is for the Perspectives Class Project, Spring 2022, at the University of Chicago. 

The [code](https://github.com/LuZhang0128/Perspectives/blob/main/Collected%20data%20and%20initial%20findings.ipynb) is written in Python 3.7 on Google Colab. <br>
All data used in this project, as well as the saved outputs, can be found [in this folder](https://drive.google.com/drive/folders/1Y1267sa7shpWpW31RBivD_CmDNQ07AeC?usp=sharing). 

# Dependencies
All the dependency (required packages) are in the `Import Packages` section in Google Colab code, and is also listed below:

```
!pip install -U git+http://github.com/UChicago-Computational-Content-Analysis/lucem_illud.git --quiet
!pip install NRCLex --quiet
!pip install -U kaleido --quiet
import pandas as pd
import re
import nltk
nltk.download('punkt')
import matplotlib.pyplot as plt
import wordcloud
import spacy
import gensim
import numpy as np 
import seaborn 
import sklearn.metrics.pairwise 
import sklearn.manifold 
import sklearn.decomposition 
import lucem_illud
import plotly.express as px
from lucem_illud.processing import normalizeTokens, trainTestSplit, word_tokenize, sent_tokenize
from nrclex import NRCLex

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
In this sampled dataset, there are in total 123517 observations. Let's get a quick look at the distribution of the data, as well as the wordclouds.
<br>

The distribution of number of tweets per day is as below: <br>
<img src="https://github.com/LuZhang0128/Perspectives/blob/main/figs/number_of_posts_per_day.png" width=60% height=60%> <br>
This distribution can be reproduced by running the `Basic Plots` section in Google Colab code. 
<br>

The wordcloud generated using all tweets in the sampled dataset as below: <br>
<img src="https://github.com/LuZhang0128/Perspectives/blob/main/figs/wordcloud_all.png" width=40% height=40%> <br>
The wordcloud generated using tweets before the death of George Floyd (2020-05-25) as below: <br>
<img src="https://github.com/LuZhang0128/Perspectives/blob/main/figs/wordcloud_before.png" width=40% height=40%> <br>
The wordcloud generated using tweets after the death of George Floyd (2020-05-25) as below: <br>
<img src="https://github.com/LuZhang0128/Perspectives/blob/main/figs/wordcloud_after.png" width=40% height=40%> <br>


# Machine Learning Model for Classification
I will train a supervised machine learning model to classify all Twitter accounts to four categorties: 
1) Social Movement Organization (SMO)
2) Other Organization
3) Social Movement Activists
4) Other Individuals

Three individual research assistants and I are currently labelling the training dataset. The manually labelled data (in progress) can be found in [this spreadsheet](https://docs.google.com/spreadsheets/d/1Re-t0Tc7OLDYzt5cJWZcztvwllyMw3mzOWhXoOYnnQM/edit?usp=sharing). One Twitter account will be at least coded by two people. Any discrepancies will be checked and resolved by a thrid people. 

The training of the machine learning model will later be updated in the Google Colab code, under the `Classification Model` section.

# Emotion Classification Before and After the Event
Since


