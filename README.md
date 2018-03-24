# Final_year_project
Topic of the project is 
Build a recommendation engine which suggests  similar products to the given product  in any e-commerce websites ex. Amazon.com, myntra.com etc 

the online course i register for  ai is 
1:https://www.appliedaicourse.com/course/apparel-recommendation-engine-workshop/

2.https://www.udemy.com/python-for-data-science-and-machine-learning-bootcamp/learn/v4/overview 

Objective of AI Work shop:
To give a flavour of what is Machine Learning/Artificial Intelligence
To introduce you how a real world machine learning problem can be solved


Python Library we are going to use is :
import numpy as np
import pandas as pd
import seaborn as sns
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity  
from sklearn.metrics import pairwise_distances
from matplotlib import gridspec

we have give a json file which consists of all information about
 the products
loading the data using pandas' read_json file.
data = pd.read_json('tops_Products.json')
