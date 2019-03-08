# Movie Recommendation Engine

Stanford CS205L WIN 2018-209 
Major course project
Creation and Analysis of Movie Recommendation Engine Using Continuous Mathematical Methods

## Index
* [Background](#background)
* [Challenges](#challenges)
* [Model](#model)
* [Data](#data)
* [References](#references)
* [Contact Info](#contact-info)
* [License](#License)

## Background
Recommendation systems are all around us in the modern world. A recommendation system is any system that attempts to predict a user's preferences and suggest a product for them to consume. Recommender systems are increasingly important for predicting users’ preferences for a variety of content including movies, books, games, products, and more. These recommender systems are a specialized subset of information filtering systems, which predict a user’s preference for a given item. The most common examples are Spotify's "Made for you" playlist, Amazon's "Recommendations for you" and "Customers who shopped for ... also shopped for ..." product suggestions, and Netflix' recommendations and "Because you watched ..." suggestions. 

While these systems have become ubiquitous with the rapid collection of massive data, they are still far from optimized. There are three main approaches to current recommender systems:

1. Collaborative Filtering - predictions based on the preferences of similar users
2. Content Based Filtering - predictions based on the preferences of the same user on similar content in the past
3. Hybrid Recommender Systems - a combination of collaborative and content based

The aim of this project is to predict a given user’s movie taste preferences based on two datasets. One dataset provides a taste profile for a subset of users, that is derived by them giving rating for movies. The other dataset is the information from other users who have watched and rated movies. Our plan is to frame the user to movie taste preference as a matrix problem. Given the high cardinality of both the  movies list and the users list, the matrix will be a sparse one (with many entries not having any ratings). This will provide a great opportunity to apply and realize some of the benefits of techniques learned in class including dimensionality reduction and other applications of SVD. To evaluate the predictiveness of our model, we plan to use the “sum of square” of errors between predicted rating and given rating as the  performance metric.

## Challenges


## Model


## Data
The data used in this project is from the [Netflix Prize](https://www.netflixprize.com/), a contest that Netflix held in 2009 that awarded $1M to the best movie recommendation algorithm based on Netflix user movie rating data. The data can be downloaded from Kaggle [here](https://www.kaggle.com/netflix-inc/netflix-prize-data). The page also offers a detailed description and brief exploration of the data.


## [References](https://github.com/polymathnexus5/rec-engine-CS205L-W19/tree/master/references)
| Source | Description |
|-------------------------------------|----------------------------------------------------------------|
| Introduction to Recommender Systems | 2016 textbook on recommender systems across industries by Charu Aggarwal |
| Recommender System Techniques Applied to Netflix Movie Data | A review of several techniques |
|The Pragmatic Theory solution to the Netflix Grand Prize| Part 1: Description of the winning recommendation system to the Netflix Prize |
| The BellKor Solution to the Netflix Grand Prize | Part 2: Description of the winning recommendation system to the Netflix Prize |
| The BigChaos Solution to the Netflix Grand Prize | Part 3: Description of the winning recommendation system to the Netflix Prize |

## Contact Info
Please reach out with any comments, questions, suggestions, ideas, or anything else.

## License
This repository contains content created by third parties, which is distributed under the license provided by those parties. Content created by Annies Abduljaffar and Matt Vail is provided under the GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007.
