# Movie Recommendation Engine

Stanford CS205L WIN 2018-2019 

Major course project:

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

This project will use a collaborative filtering approach to predict a given user’s movie taste preferences based on their movie reviews and the reviews of other users. We will frame the movie recommendation problem as a matrix problem. Given the high cardinality of both the  movies list and the users list, the matrix will be a sparse one (with many entries not having any ratings). This will provide a great opportunity to apply and realize some of the benefits of techniques learned in class including dimensionality reduction and other applications of SVD. To evaluate the predictiveness of our model, we plan to use the “sum of square” of errors between predicted rating and given rating as the  performance metric.

## Challenges


## Model


## Data

The data used in this project is from MovieLens. The data can be downloaded [here](http://files.grouplens.org/datasets/movielens/ml-20m.zip). This [page](http://files.grouplens.org/datasets/movielens/ml-20m-README.html) offers a detailed description of the data.


## [References](https://github.com/polymathnexus5/rec-engine-CS205L-W19/tree/master/references)
| Source | Description |
|-------------------------------------|----------------------------------------------------------------|
| Introduction to Recommender Systems | 2016 textbook on recommender systems across industries by Charu Aggarwal |
| Recommender System Techniques Applied to Netflix Movie Data | A review of several techniques |
|The Pragmatic Theory solution to the Netflix Grand Prize| Part 1: Description of the winning recommendation system to the Netflix Prize |
| The BellKor Solution to the Netflix Grand Prize | Part 2: Description of the winning recommendation system to the Netflix Prize |
| The BigChaos Solution to the Netflix Grand Prize | Part 3: Description of the winning recommendation system to the Netflix Prize |
| Movie Recommender System | Comparison of movie recommender systems trained on the MovieLens dataset |
| Dimensionality Reduction SVD and Its Applications | Educational presentation on SVD and some of its applications |

Other references:
* [Collaborative Filtering](https://en.wikipedia.org/wiki/Collaborative_filtering)
* [Recommender Systems](https://en.wikipedia.org/wiki/Recommender_system)
* [Information Filtering Systems](https://en.wikipedia.org/wiki/Information_filtering_system)
* [Movie Recommender System on Kaggle](https://www.kaggle.com/rounakbanik/movie-recommender-systems)

## Contact Info
Please reach out with any comments, questions, suggestions, ideas, or anything else.

## License
This repository contains content created by third parties, which is distributed under the license provided by those parties. Content created by Annies Abduljaffar and Matt Vail is provided under the GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007.
