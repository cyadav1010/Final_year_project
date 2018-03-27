import json
import pandas as pd
from collections import Counter
from pprint import pprint
import itertools
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import pickle

#data=json.load(open('/home/chandan/Downloads/text.json'))
data = pd.read_json('/home/chandan/Downloads/AI/tops_fashion.json')
#data = pd.read_json('/home/chandan/Downloads/AI/text.json')
#data1=pd.read_json('/home/chandan/Downloads/data.json')
pprint(data)
#print ('Number of data points : ', info.shape[0]);
#print('Number of features/variables:', data.shape[1]);
print(data.columns);
print ('Number of data points : ', data.shape[0], \
       'Number of features/variables:', data.shape[1])
data = data[['asin', 'brand', 'color', 'medium_image_url', 'product_type_name', 'title', 'formatted_price']]
print ('Number of data points : ', data.shape[0], \
       'Number of features:', data.shape[1])
data.head()
print(data['product_type_name'].describe())
print(data['product_type_name'].unique())
product_type_count = Counter(list(data['product_type_name']))
print(product_type_count.most_common(10))
print(data['brand'].describe())
brand_count = Counter(list(data['brand']))
print(brand_count.most_common(10))

print(data['color'].describe())
color_count = Counter(list(data['color']))
print(color_count.most_common(10))
print(data['formatted_price'].describe())
price_count = Counter(list(data['formatted_price']))
print(price_count.most_common(10))
print(data['title'].describe())
data.to_pickle('/home/chandan/Downloads/AI/text.pkl')
data = data.loc[~data['formatted_price'].isnull()]
print('Number of data points After eliminating price=NULL :', data.shape[0])
data =data.loc[~data['color'].isnull()]
print('Number of data points After eliminating color=NULL :', data.shape[0])
data.to_pickle('/home/chandan/Downloads/AI/28k_apparel_data')
print(sum(data.duplicated('title')))
data = pd.read_pickle('/home/chandan/Downloads/AI/28k_apparel_data')
print(data.head())
data_sorted = data[data['title'].apply(lambda x: len(x.split())>4)]
print("After removal of products with short description:", data_sorted.shape[0])
stage1_dedupe_asins = []
i = 0
j = 0
num_data_points = data_sorted.shape[0]
while i < num_data_points and j < num_data_points:
       previous_i = i
       # store the list of words of ith string in a, ex: a = ['tokidoki', 'The', 'Queen', 'of', 'Diamonds', 'Women's', 'Shirt', 'X-Large']
       a = data['title'].loc[indices[i]].split(
       # search for the similar products sequentially
       j = i + 1
       while j < num_data_points:
              # store the list of words of jth string in b, ex: b = ['tokidoki', 'The', 'Queen', 'of', 'Diamonds', 'Women's', 'Shirt', 'Small']
              b = data['title'].loc[indices[j]].split()
              # store the maximum length of two strings
              length = max(len(a), len(b))
              # count is used to store the number of words that are matched in both strings
              count = 0
              # itertools.zip_longest(a,b): will map the corresponding words in both strings, it will appened None in case of unequal strings
              # example: a =['a', 'b', 'c', 'd']
              # b = ['a', 'b', 'd']
              # itertools.zip_longest(a,b): will give [('a','a'), ('b','b'), ('c','d'), ('d', None)]
              for k in itertools.zip_longest(a, b):
                     if (k[0] == k[1]):
                            count += 1
              # if the number of words in which both strings differ are > 2 , we are considering it as those two apperals are different
              # if the number of words in which both strings differ are < 2 , we are considering it as those two apperals are same, hence we are ignoring them
              if (length - count) > 2:  # number of words in which both sensences differ
                     # if both strings are differ by more than 2 words we include the 1st string index
                     stage1_dedupe_asins.append(data_sorted['asin'].loc[indices[i]])
                     # if the comaprision between is between num_data_points, num_data_points-1 strings and they differ in more than 2 words we include both
                     if j == num_data_points - 1: stage1_dedupe_asins.append(data_sorted['asin'].loc[indices[j]])
                     # start searching for similar apperals corresponds 2nd string
                     i = j
                     break
              else:
                     j += 1
       if previous_i == i:
              break
# This code snippet takes significant amount of time.
# O(n^2) time.
# Takes about an hour to run on a decent computer.
indices = []
for i, row in data.iterrows():
       indices.append(i)
stage2_dedupe_asins = []
while len(indices) != 0:
       i = indices.pop()
       stage2_dedupe_asins.append(data['asin'].loc[i])
       # consider the first apperal's title
       a = data['title'].loc[i].split()
       # store the list of words of ith string in a, ex: a = ['tokidoki', 'The', 'Queen', 'of', 'Diamonds', 'Women's', 'Shirt', 'X-Large']
       for j in indices:
              b = data['title'].loc[j].split()
              # store the list of words of jth string in b, ex: b = ['tokidoki', 'The', 'Queen', 'of', 'Diamonds', 'Women's', 'Shirt', 'X-Large']
              length = max(len(a), len(b))
              # count is used to store the number of words that are matched in both strings
              count = 0
              # itertools.zip_longest(a,b): will map the corresponding words in both strings, it will appened None in case of unequal strings
              # example: a =['a', 'b', 'c', 'd']
              # b = ['a', 'b', 'd']
              # itertools.zip_longest(a,b): will give [('a','a'), ('b','b'), ('c','d'), ('d', None)]
              for k in itertools.zip_longest(a, b):
                     if (k[0] == k[1]):
                            count += 1
              # if the number of words in which both strings differ are < 3 , we are considering it as those two apperals are same, hence we are ignoring them
              if (length - count) < 3:
                     indices.remove(j)
# we use the list of stop words that are downloaded from nltk lib.
stop_words = set(stopwords.words('english'))
print ('list of stop words:', stop_words)

def nlp_preprocessing(total_text, index, column):
    if type(total_text) is not int:
        string = ""
        for words in total_text.split():
            # remove the special chars in review like '"#$@!%^&*()_+-~?>< etc.
            word = ("".join(e for e in words if e.isalnum()))
            # Conver all letters to lower-case
            word = word.lower()
            # stop-word removal
            if not word in stop_words:
                string += word + " "
        data[column][index] = string
# Utility Functions which we will use through the rest of the workshop.
# Display an image
def display_img(url, ax, fig):
       # we get the url of the apparel and download it
       response = requests.get(url)
       img = Image.open(BytesIO(response.content))
       # we will display it in notebook
       plt.imshow(img)
# plotting code to understand the algorithm's decision.
def plot_heatmap(keys, values, labels, url, text):
       # keys: list of words of recommended title
       # values: len(values) ==  len(keys), values(i) represents the occurence of the word keys(i)
       # labels: len(labels) == len(keys), the values of labels depends on the model we are using
       # if model == 'bag of words': labels(i) = values(i)
       # if model == 'tfidf weighted bag of words':labels(i) = tfidf(keys(i))
       # if model == 'idf weighted bag of words':labels(i) = idf(keys(i))
       # url : apparel's url
       # we will devide the whole figure into two parts
       gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[4, 1])
       fig = plt.figure(figsize=(25, 3))
       # 1st, ploting heat map that represents the count of commonly ocurred words in title2
       ax = plt.subplot(gs[0])
       # it displays a cell in white color if the word is intersection(lis of words of title1 and list of words of title2), in black if not
       ax = sns.heatmap(np.array([values]), annot=np.array([labels]))
       ax.set_xticklabels(keys)  # set that axis labels as the words of title
       ax.set_title(text)  # apparel title
       # 2nd, plotting image of the the apparel
       ax = plt.subplot(gs[1])
       # we don't want any grid lines for image and no labels on x-axis and y-axis
       ax.grid(False)
       ax.set_xticks([])
       ax.set_yticks([])
       # we call dispaly_img based with paramete url
       display_img(url, ax, fig)
       # displays combine figure ( heat map and image together)
       plt.show()
def plot_heatmap_image(doc_id, vec1, vec2, url, text, model):
       # doc_id : index of the title1
       # vec1 : input apparels's vector, it is of a dict type {word:count}
       # vec2 : recommended apparels's vector, it is of a dict type {word:count}
       # url : apparels image url
       # text: title of recomonded apparel (used to keep title of image)
       # model, it can be any of the models,
       # 1. bag_of_words
       # 2. tfidf
       # 3. idf
       # we find the common words in both titles, because these only words contribute to the distance between two title vec's
       intersection = set(vec1.keys()) & set(vec2.keys())
       # we set the values of non intersecting words to zero, this is just to show the difference in heatmap
       for i in vec2:
              if i not in intersection:
                     vec2[i] = 0
       # for labeling heatmap, keys contains list of all words in title2
       keys = list(vec2.keys())
       #  if ith word in intersection(lis of words of title1 and list of words of title2): values(i)=count of that word in title2 else values(i)=0
       values = [vec2[x] for x in vec2.keys()]
       # labels: len(labels) == len(keys), the values of labels depends on the model we are using
       # if model == 'bag of words': labels(i) = values(i)
       # if model == 'tfidf weighted bag of words':labels(i) = tfidf(keys(i))
       # if model == 'idf weighted bag of words':labels(i) = idf(keys(i))
       if model == 'bag_of_words':
              labels = values
       elif model == 'tfidf':
              labels = []
              for x in vec2.keys():
                     # tfidf_title_vectorizer.vocabulary_ it contains all the words in the corpus
                     # tfidf_title_features[doc_id, index_of_word_in_corpus] will give the tfidf value of word in given document (doc_id)
                     if x in tfidf_title_vectorizer.vocabulary_:
                            labels.append(tfidf_title_features[doc_id, tfidf_title_vectorizer.vocabulary_[x]])
                     else:
                            labels.append(0)
       elif model == 'idf':
              labels = []
              for x in vec2.keys():
                     # idf_title_vectorizer.vocabulary_ it contains all the words in the corpus
                     # idf_title_features[doc_id, index_of_word_in_corpus] will give the idf value of word in given document (doc_id)
                     if x in idf_title_vectorizer.vocabulary_:
                            labels.append(idf_title_features[doc_id, idf_title_vectorizer.vocabulary_[x]])
                     else:
                            labels.append(0)
       plot_heatmap(keys, values, labels, url, text)
# this function gets a list of wrods along with the frequency of each
# word given "text"
def text_to_vector(text):
       word = re.compile(r'\w+')
       words = word.findall(text)
       # words stores list of all words in given string, you can try 'words = text.split()' this will also gives same result
       return Counter(
              words)  # Counter counts the occurence of each word in list, it returns dict type object {word1:count}
def get_result(doc_id, content_a, content_b, url, model):
       text1 = content_a
       text2 = content_b
       # vector1 = dict{word11:#count, word12:#count, etc.}
       vector1 = text_to_vector(text1)
       # vector1 = dict{word21:#count, word22:#count, etc.}
       vector2 = text_to_vector(text2)
       plot_heatmap_image(doc_id, vector1, vector2, url, text2, model)
title_vectorizer = CountVectorizer()
title_features   = title_vectorizer.fit_transform(data['title'])
title_features.get_shape() # get number of rows and columns in feature matrix.
# title_features.shape = #data_points * #words_in_corpus
# CountVectorizer().fit_transform(corpus) returns
# the a sparase matrix of dimensions #data_points * #words_in_corpus
# What is a sparse vector?
# title_features[doc_id, index_of_word_in_corpus] = number of times the word occured in that doc
def bag_of_words_model(doc_id, num_results):
       # doc_id: apparel's id in given corpus

       # pairwise_dist will store the distance from given input apparel to all remaining apparels
       # the metric we used here is cosine, the coside distance is mesured as K(X, Y) = <X, Y> / (||X||*||Y||)
       # http://scikit-learn.org/stable/modules/metrics.html#cosine-similarity
       pairwise_dist = pairwise_distances(title_features, title_features[doc_id])
       # np.argsort will return indices of the smallest distances
       indices = np.argsort(pairwise_dist.flatten())[0:num_results]
       # pdists will store the smallest distances
       pdists = np.sort(pairwise_dist.flatten())[0:num_results]
       # data frame indices of the 9 smallest distace's
       df_indices = list(data.index[indices])
       for i in range(0, len(indices)):
              # we will pass 1. doc_id, 2. title1, 3. title2, url, model
              get_result(indices[i], data['title'].loc[df_indices[0]], data['title'].loc[df_indices[i]],
                         data['medium_image_url'].loc[df_indices[i]], 'bag_of_words')
              print('ASIN :', data['asin'].loc[df_indices[i]])
              print('Brand:', data['brand'].loc[df_indices[i]])
              print('Title:', data['title'].loc[df_indices[i]])
              print('Euclidean similarity with the query image :', pdists[i])
              print('=' * 60)
# call the bag-of-words model for a product to get similar products.
bag_of_words_model(12566, 20)  # change the index if you want to.
# In the output heat map each value represents the count value
# of the label word, the color represents the intersection
# with inputs title.
# try 12566
# try 931
tfidf_title_vectorizer = TfidfVectorizer(min_df = 0)
tfidf_title_features = tfidf_title_vectorizer.fit_transform(data['title'])
# tfidf_title_features.shape = #data_points * #words_in_corpus
# CountVectorizer().fit_transform(courpus) returns the a sparase matrix of dimensions #data_points * #words_in_corpus
# tfidf_title_features[doc_id, index_of_word_in_corpus] = tfidf values of the word in given doc
def tfidf_model(doc_id, num_results):
       # doc_id: apparel's id in given corpus
       # pairwise_dist will store the distance from given input apparel to all remaining apparels
       # the metric we used here is cosine, the coside distance is mesured as K(X, Y) = <X, Y> / (||X||*||Y||)
       # http://scikit-learn.org/stable/modules/metrics.html#cosine-similarity
       pairwise_dist = pairwise_distances(tfidf_title_features, tfidf_title_features[doc_id])

       # np.argsort will return indices of 9 smallest distances
       indices = np.argsort(pairwise_dist.flatten())[0:num_results]
       # pdists will store the 9 smallest distances
       pdists = np.sort(pairwise_dist.flatten())[0:num_results]
       # data frame indices of the 9 smallest distace's
       df_indices = list(data.index[indices])
       for i in range(0, len(indices)):
              # we will pass 1. doc_id, 2. title1, 3. title2, url, model
              get_result(indices[i], data['title'].loc[df_indices[0]], data['title'].loc[df_indices[i]],
                         data['medium_image_url'].loc[df_indices[i]], 'tfidf')
              print('ASIN :', data['asin'].loc[df_indices[i]])
              print('BRAND :', data['brand'].loc[df_indices[i]])
              print('Eucliden distance from the given image :', pdists[i])
              print('=' * 125)
tfidf_model(12566, 20)
# in the output heat map each value represents the tfidf values of the label word, the color represents the intersection with inputs title
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import pickle

# in this project we are using a pretrained model by google
# its 3.3G file, once you load this into your memory
# it occupies ~9Gb, so please do this step only if you have >12G of ram
# we will provide a pickle file wich contains a dict ,
# and it contains all our courpus words as keys and  model[word] as values
# To use this code-snippet, download "GoogleNews-vectors-negative300.bin"
# from https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit
# it's 1.9GB in size.

'''
model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
'''

#if you do NOT have RAM >= 12GB, use the code below.
with open('word2vec_model', 'rb') as handle:
    model = pickle.load(handle)


# Utility functions

def get_word_vec(sentence, doc_id, m_name):
       # sentence : title of the apparel
       # doc_id: document id in our corpus
       # m_name: model information it will take two values
       # if  m_name == 'avg', we will append the model[i], w2v representation of word i
       # if m_name == 'weighted', we will multiply each w2v[word] with the idf(word)
       vec = []
       for i in sentence.split():
              if i in vocab:
                     if m_name == 'weighted' and i in idf_title_vectorizer.vocabulary_:
                            vec.append(idf_title_features[doc_id, idf_title_vectorizer.vocabulary_[i]] * model[i])
                     elif m_name == 'avg':
                            vec.append(model[i])
              else:
                     # if the word in our courpus is not there in the google word2vec corpus, we are just ignoring it
                     vec.append(np.zeros(shape=(300,)))
       # we will return a numpy array of shape (#number of words in title * 300 ) 300 = len(w2v_model[word])
       # each row represents the word2vec representation of each word (weighted/avg) in given sentance
       return np.array(vec)


def get_distance(vec1, vec2):
       # vec1 = np.array(#number_of_words_title1 * 300), each row is a vector of length 300 corresponds to each word in give title
       # vec2 = np.array(#number_of_words_title2 * 300), each row is a vector of length 300 corresponds to each word in give title

       final_dist = []
       # for each vector in vec1 we caluclate the distance(euclidean) to all vectors in vec2
       for i in vec1:
              dist = []
              for j in vec2:
                     # np.linalg.norm(i-j) will result the euclidean distance between vectors i, j
                     dist.append(np.linalg.norm(i - j))
              final_dist.append(np.array(dist))
       # final_dist = np.array(#number of words in title1 * #number of words in title2)
       # final_dist[i,j] = euclidean distance between vectors i, j
       return np.array(final_dist)


def heat_map_w2v(sentence1, sentence2, url, doc_id1, doc_id2, model):
       # sentance1 : title1, input apparel
       # sentance2 : title2, recommended apparel
       # url: apparel image url
       # doc_id1: document id of input apparel
       # doc_id2: document id of recommended apparel
       # model: it can have two values, 1. avg 2. weighted
       # s1_vec = np.array(#number_of_words_title1 * 300), each row is a vector(weighted/avg) of length 300 corresponds to each word in give title
       s1_vec = get_word_vec(sentence1, doc_id1, model)
       # s2_vec = np.array(#number_of_words_title1 * 300), each row is a vector(weighted/avg) of length 300 corresponds to each word in give title
       s2_vec = get_word_vec(sentence2, doc_id2, model)
       # s1_s2_dist = np.array(#number of words in title1 * #number of words in title2)
       # s1_s2_dist[i,j] = euclidean distance between words i, j
       s1_s2_dist = get_distance(s1_vec, s2_vec)
       # devide whole figure into 2 parts 1st part displays heatmap 2nd part displays image of apparel
       gs = gridspec.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[2, 1])
       fig = plt.figure(figsize=(15, 15))
       ax = plt.subplot(gs[0])
       # ploting the heap map based on the pairwise distances
       ax = sns.heatmap(np.round(s1_s2_dist, 4), annot=True)
       # set the x axis labels as recommended apparels title
       ax.set_xticklabels(sentence2.split())
       # set the y axis labels as input apparels title
       ax.set_yticklabels(sentence1.split())
       # set title as recommended apparels title
       ax.set_title(sentence2)
       ax = plt.subplot(gs[1])
       # we remove all grids and axis labels for image
       ax.grid(False)
       ax.set_xticks([])
       ax.set_yticks([])
       display_img(url, ax, fig)
       plt.show()
# vocab = stores all the words that are there in google w2v model
# vocab = model.wv.vocab.keys() # if you are using Google word2Vec
vocab = model.keys()
# this function will add the vectors of each word and returns the avg vector of given sentance
def build_avg_vec(sentence, num_features, doc_id, m_name):
       # sentace: its title of the apparel
       # num_features: the lenght of word2vec vector, its values = 300
       # m_name: model information it will take two values
       # if  m_name == 'avg', we will append the model[i], w2v representation of word i
       # if m_name == 'weighted', we will multiply each w2v[word] with the idf(word)
       featureVec = np.zeros((num_features,), dtype="float32")
       # we will intialize a vector of size 300 with all zeros
       # we add each word2vec(wordi) to this fetureVec
       nwords = 0
       for word in sentence.split():
              nwords += 1
              if word in vocab:
                     if m_name == 'weighted' and word in idf_title_vectorizer.vocabulary_:
                            featureVec = np.add(featureVec,
                                                idf_title_features[doc_id, idf_title_vectorizer.vocabulary_[word]] *
                                                model[word])
                     elif m_name == 'avg':
                            featureVec = np.add(featureVec, model[word])
       if (nwords > 0):
              featureVec = np.divide(featureVec, nwords)
       # returns the avg vector of given sentance, its of shape (1, 300)
       return featureVec
doc_id = 0
w2v_title = []
# for every title we build a avg vector representation
for i in data['title']:
    w2v_title.append(build_avg_vec(i, 300, doc_id,'avg'))
    doc_id += 1
# w2v_title = np.array(# number of doc in courpus * 300), each row corresponds to a doc
w2v_title = np.array(w2v_title)
