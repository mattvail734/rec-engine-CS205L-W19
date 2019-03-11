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
* [Software](#software)
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

The primary challenge is that the user x ratings matrix (A) is very sparse. Only ~0.5% of all entries are non-zero, but the total size of the matrix is > 60GB. In order to overcome this challenge, we will try to do all the required computation using matrix  methods that avoid using the complete ratings matrix and allow us to compute one column or row vector at a time.

Note: iterative methods (i.e. gradient descent, conjugate GD, etc.) for preconditioning (HW4, Heath 11.5.5, 

## Model


## Data

The data used in this project is from MovieLens. The data can be downloaded [here](http://files.grouplens.org/datasets/movielens/ml-20m.zip). This [page](http://files.grouplens.org/datasets/movielens/ml-20m-README.html) offers a detailed description of the data.


## Software

The entire project was completed in Python 3.7.2_1 using standard libraries including:

* [Os](https://docs.python.org/3/library/os.html)
* [SciKitLearn](https://scikit-learn.org/stable/)
* [Numpy](http://www.numpy.org/)
* [Pandas](http://pandas.pydata.org/)
* [Matplotlib](https://matplotlib.org/)
* [Seaborn](https://seaborn.pydata.org/)

We also used the [Surprise](https://surprise.readthedocs.io/en/stable/index.html) library for benchmarks.

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
| Deep Item Based Collaborative Filtering for Sparse Implicit Feedback | Recent arXiv paper on deep learning for recommmender systems focused on implicit feedback but very relevant for details on sparse matrices |
| Factor in the Neighbors: Scalable and Accurate Collaborative Filtering | Yehuda Koren research paper on collaborative filtering with a good explanation of the underlying mathematics |
| A Scalable Collaborative Filtering Framework based on Co-clustering | UT paper on clustering and collaborative filtering |
| Factorization Meets the Neighborhood: a Multifaceted Collaborative Filtering Model | Yehuda Koren paper on neighborhood model based collaborative filtering |
| Matrix Factorization Techniques for Recommender Systems | Yehuda Koren paper on matrix factorization methods for collaborative filtering |
| Algorithms for Non-negative Matrix Factorization | Non-negative matrix factorization |
| Slope One Predictors for Online Rating-Based Collaborative Filtering | Online models |
| Probabilistic Matrix Factorization | Probabilistic approach to matrix factorization |
| Learning from Incomplete Ratings Using Non-negative Matrix Factorization | Non-negative matrix factorization |

Other references:
* [Collaborative Filtering Wiki](https://en.wikipedia.org/wiki/Collaborative_filtering)
* [Recommender Systems Wiki](https://en.wikipedia.org/wiki/Recommender_system)
* [Information Filtering Systems Wiki](https://en.wikipedia.org/wiki/Information_filtering_system)
* [Pearson Correlation Coefficient Wiki](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)
* [Movie Recommender System on Kaggle](https://www.kaggle.com/rounakbanik/movie-recommender-systems)
* [Gradient Descent Based Matrix Factorization Approach](https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-recommendation-engine-python/)
* [Interesting Mathematical Details from a Netflix Prize Finalize](https://sifter.org/~simon/journal/20061211.html)

## Contact Info
Please reach out with any comments, questions, suggestions, ideas, or anything else.

## License
This repository contains content created by third parties, which is distributed under the license provided by those parties. Content created by Annies Abduljaffar and Matt Vail is provided under the GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007.
