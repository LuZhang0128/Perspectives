# Perspectives
The code and data in this repository is for the Perspectives Class Project, Spring 2022, at the University of Chicago. 

The [code](https://github.com/LuZhang0128/Perspectives/blob/main/Collected%20data%20and%20initial%20findings.ipynb) is written in Python 3.7 on Google Colab. 
All data used in this project, as well as the saved outputs, can be found [here](https://drive.google.com/drive/folders/1Y1267sa7shpWpW31RBivD_CmDNQ07AeC?usp=sharing). 

# Dependencies
All the dependency (required packages) are in the `Import Packages` section, and is also listed below:

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
Here I've collaborated with my friend and have written a scraper for Twitter data. However, it only randomly send back a certain amount of Tweets per day every time I try to make request through the code. For this class project, I just used one output file, which can be found [here](https://drive.google.com/file/d/19umvDYuu1o6uIr3xz1m0qwLay4qgeQ5r/view?usp=sharing). This project can expand the scale by running with multiple output files in the data folder. 

You can pre-process the data by running the `Load and Clean Data` section in Google Colab code. Here removed urls and uninformative information like websites and '@'s. I then splited data into before and after the death of George Floyd. I then tokenized and normalized all tokens. 

The pre-processed data can be found here. 





