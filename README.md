[![DOI](https://zenodo.org/badge/485132738.svg)](https://zenodo.org/badge/latestdoi/485132738)

# Overview
The code and data in this repository is for the Perspectives Class Project, Spring 2022, at the University of Chicago. 

The purpose of this class project is to evaluate the Fringe Effect of Social Movement Organizations (SMOs) in Online Social Movements. More specifically, I want to study the BlackLivesMatter movement, and see how SMOs' popularies and display of emotion changed after the death of George Floyd.

The [code](https://github.com/LuZhang0128/Perspectives/blob/main/analysis.ipynb) is written in Python 3.7 on Google Colab. <br>
All data used in this project, as well as the saved outputs, can be found [in this folder](https://drive.google.com/drive/folders/1Y1267sa7shpWpW31RBivD_CmDNQ07AeC?usp=sharing). 

# Dependencies
All the dependency (required packages) are in the `Import Packages` section in Google Colab code, and are also listed in the [package_dependencies.rtf](https://github.com/LuZhang0128/Perspectives/blob/main/package_dependencies.rtf). Note that when running the Google Colab code, there will be warnings and errors due to internal conflicts within packages. This conflict would not impact further codes and analysis. 

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
In this sampled dataset, there are in total 123517 observations. Let's get a quick look at the distribution of the data, as well as the wordclouds. The distribution and word clouds can be reproduced by running the `Basic Plots` section in Google Colab code. Note that this basic description only get people to know the dataset better, and is not related to the final answer to the research question.
<br>

### Distribution
The distribution of number of tweets per day is as below: <br>
<img src="https://github.com/LuZhang0128/Perspectives/blob/main/figs/number_of_posts_per_day.png" width=60% height=60%>
<br>

This plot, however, does not necessarily reflect the true distribution of posts. After 2016, the number of posts is more likely to be limited by the algorithm instead of telling us the true trend. This limitation, on the other hand, empirically demonstrated the statement that the BlackLivesMatter movement became global only after 2016. If we want to study the trend, [Google Trend](https://trends.google.com/trends/explore?date=2007-12-31%202022-04-24&geo=US&q=blacklivesmatter) is an alternative source. <br>

### WordClouds
The wordcloud generated using all tweets in the sampled dataset as below: <br>
<img src="https://github.com/LuZhang0128/Perspectives/blob/main/figs/wordcloud_all.png" width=40% height=40%> <br>
The wordcloud generated using tweets before (right) and after (left)the death of George Floyd (2020-05-25) are as below: <br>
<img src="https://github.com/LuZhang0128/Perspectives/blob/main/figs/wordcloud_before.png" width=40% height=40%> 
<img src="https://github.com/LuZhang0128/Perspectives/blob/main/figs/wordcloud_after.png" width=40% height=40%> 
<br>

It is interesting to see words such as "wildhorse," "wild," "horse," and "land" in the first word cloud, and not in the second one. This shift indicates a slight change in topic before and after the identified event (death of George Floyd). It also shows that I'm on the right track.


# Machine Learning Model for Classification
I trained four supervised machine learning models to classify all Twitter accounts to four categorties using grid-serach for hyper parameter tuning: 
1) Social Movement Organization (SMO)
2) Other Organization
3) Social Movement Activists
4) Other Individuals

Three individual research assistants and I are currently labelling the training dataset. The manually labelled data (in progress) can be found in [this spreadsheet](https://docs.google.com/spreadsheets/d/1Re-t0Tc7OLDYzt5cJWZcztvwllyMw3mzOWhXoOYnnQM/edit?usp=sharing). One Twitter account is at least coded by two people. Any discrepancies are checked and resolved by a thrid people. <br>

The distribution of the manually labeled data is plotted in R-studio using [this code](https://github.com/LuZhang0128/Perspectives/blob/main/label_distribution.Rmd) and is as follows: <br>
<img src="https://github.com/LuZhang0128/Perspectives/blob/main/figs/labeled_distribution.png" width=60% height=60%> 
<br>

There are 55 (12%) Social Movement Organizations, 67 (15%) Other Organizations, 84 (19%) Social Movement Activists, and 244 (54%) Other Individuals. Most accounts are everyday people. These 450 accounts are responsible for 3,566 (2.89%) tweets among all 123,517 tweets. <br>

The trainings and hyperparameter tuning of the four machine learning model (Random Forest Classifier, Logistic Regression, Support Vector Classification (SVC), and Multinomial Na??ve Bayes algorithm) are in the Google Colab code, under the `Machine Learning Model` section. <br>

The Confusion Matrices are of the models are: <br>
<img src="https://github.com/LuZhang0128/Perspectives/blob/main/figs/confusion_matricies.png" width=60% height=60%> <br>
and the accuracy scores are: <br>
<img src="https://github.com/LuZhang0128/Perspectives/blob/main/figs/accuracy_scores.png" width=60% height=60%> <br>

The confusion matrices show that all four models experience high errors when differentiating SMOs??? accounts from Individual Activists??? accounts. When the true label is 3 (Social Movement Activists), the models are more likely to falsely classify it as SMO than the other two categories. Meanwhile, the models can tell Individual Activists apart from everyday people. When the true label is 4 (Other Individual), the model is doing a good job doing the classification. The error rates of classifying them into other three categories are similar, meaning that the model did not find significantly higher seminaries between everyday people and individual activists. <br>

Based on the accuracy scores, the Random Forest classifier achieved the highest accuracy score. However, all four models have better performance on the training sets compared to the testing sets. This suggests over-fitting of the models. The accuracy scores on the testing sets are around 70%. <br>

# Emotion Classification
Due to the over-fitting problem, the classification of the rest of the accounts would be inaccurate. Thus, I decided not to delve down to examine the behavior patterns of each type of account until more accounts are manually labeled. In this study, I would only look at the general emotion trends in the public discursive field before and after the death of George Floyd using all tweets. <br>

I implemented a roBERTa-base model pre-trained on around 58 million tweets and finetuned for emotion recognition with TweetEval benchmark. This model is suitable for labeling tweets with four different emotions: Joy, Sadness, Optimism, and Anger. The model returns a score for each emotion, with 1 meaning the highest score and 0 meaning the lowest score. I then calculate an average score for each day. The implementaion code and the regressions that will be discussed later can be found in the the Google Colab code under the 'Pre-trained BERT model for Emotion Classification' section. <br>

Based on data distribution figure and the discussions above, the BlackLivesMatter movement did not get enough attention from the public before 2016. The following graph also shows that the emotional scores before 2016 are messy, especially for joy and anger, aligning with the previous statement. Thus, I decided to focus only on data after 2016. I performed two separate linear regressions for each of the four emotions before and after the death of George Floyd. <br>

<img src="https://github.com/LuZhang0128/Perspectives/blob/main/figs/average_joy_score_per_day.png.png" width=40% height=40%> <img src="https://github.com/LuZhang0128/Perspectives/blob/main/figs/average_optimism_score_per_day.png.png" width=40% height=40%> <br>
<img src="https://github.com/LuZhang0128/Perspectives/blob/main/figs/average_anger_score_per_day.png" width=40% height=40%> <img src="https://github.com/LuZhang0128/Perspectives/blob/main/figs/average_sadness_score_per_day.png.png" width=40% height=40%> <br>
<br>

There are observed significant jump discontinuities between the regressions. For joy and optimism, there is a sudden increase in scores after the event. The scores gradually decrease almost back to the original level. Similarly, there is a sharp drop in scores for anger and sadness after the event, and the scores gradually increase almost back to the original level. The regression lines' slopes after the event are steeper than those before the event. <br>

# Emotion and Popularity
There I used the number of Likes for each post as a proxy of the popularity of that post to test the relationship between emotion and tweets' popularity. You can run the `Popularity and Emotion` section in Google Colab to replicate the anlaysis. The distribution of the number of Likes and the zoomed-in visualization are presented below: <br>
<img src="https://github.com/LuZhang0128/Perspectives/blob/main/figs/dis_num_like.png" width=60% height=60%> <br>

Out of the 123,517 tweets in the dataset, 113,383 tweets (91.80%) received less than 100 Likes, with a mean of 1,174 Likes and a median of 2 Likes. 37,122 tweets (30.05%) received 0 Likes. Since the data is right-skewed, I applied the log(x+1) transformation on the data to achieve a relatively normalized data for later Ordinary Least Square (OLS) linear regressions. <br>

I performed OLS linear regressions between the number of Likes and each of the four emotions before and after the identified event (George Floyd???s death). The visualizations are presented below: <br>
<img src="https://github.com/LuZhang0128/Perspectives/blob/main/figs/joy_num_like.png" width=40% height=40%> <img src="https://github.com/LuZhang0128/Perspectives/blob/main/figs/optimism_num_like.png" width=40% height=40%> <br>
<img src="https://github.com/LuZhang0128/Perspectives/blob/main/figs/anger_num_like.png" width=40% height=40%> <img src="https://github.com/LuZhang0128/Perspectives/blob/main/figs/sadness_num_like.png" width=40% height=40%> <br>
<br>

Detailed statistics of the regression models are: <br>
<img src="https://github.com/LuZhang0128/Perspectives/blob/main/figs/table.png" width=60% height=60%> <br>

The regressions show a statistically significant negative correlation between Joy and Optimism scores and the number of Likes (after the log(x+1) transformation) for tweets. Meanwhile, there is a statistically significant positive correlation between Anger and Sadness scores and the number of Likes (after the log(x+1) transformation) for tweets. The absolute numbers of coefficients after the identified event are larger than those before the event, suggesting that a unit increase in the displayed emotion can lead to a greater number of Likes after the event. In other words, extreme emotion can attract more attention after George Floyd???s death than before.

# Cite this Repository
You can cite this repository via the doi on the very top of the README file.

