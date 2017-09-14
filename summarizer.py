import glob
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
from extractive import summarize
from preprocess import pre_process
from summary import summary_sents


#Read files
path = raw_input('Please Enter Directory Path: ')
print 'Reading files...'
path = path+"*.txt"
files = glob.glob(path)
docs = []
for name in files: 
    with open(name) as f: 
        docs.append(f.read())

#add title as headline
fname = [fname[12:] for fname in files]
fname = [re.sub(".txt",". ",n) for n in fname]
docs = map(str.__add__,fname,docs)

print 'Clustering Documents...'

#tf-idf
tfidf_vectorizer = TfidfVectorizer(max_df=0.8,max_features=200000,decode_error='replace',
                                 min_df=0.2, stop_words='english',
                                 use_idf=True, tokenizer=pre_process, ngram_range=(1,3))

tfidf_matrix = tfidf_vectorizer.fit_transform(docs)
terms = tfidf_vectorizer.get_feature_names()

#Clustering
corpus_similarity = cosine_similarity(tfidf_matrix)#Similarity
km = KMeans(n_clusters=3)
km.fit(tfidf_matrix)
clusters = km.labels_.tolist()

print 'Modelling Topics...'

#topic extraction
lda = LatentDirichletAllocation(n_topics=3,max_iter=200,learning_method='online',
                                learning_offset=50.,random_state=42)
lda.fit(tfidf_matrix)

topics=[]
for topic_idx, topic in enumerate(lda.components_):
    topics.append(" ".join([terms[i]
                        for i in topic.argsort()[:-30 - 1:-1]]))



#finding similar topic
query = raw_input('Please Enter Topic: ')
topics.append(query)
tfidf_topics_matrix = tfidf_vectorizer.fit_transform(topics)
topic_similarity = cosine_similarity(tfidf_topics_matrix)
topics=topics[:-1]

#index of most similar cluster
similar_clust = np.argmax(topic_similarity[3][:3])
article_indices = [i for i, x in enumerate(clusters) if x == similar_clust]

print 'Building Extractive Summary...'

#extractive summary
ext_summary = []
for i in article_indices:
    ext_summary.append(summarize(docs[i]))


#subtopic relevant summary
sub_topic=[]
sub_topic.append(raw_input('Please Enter Sub Topic 01: '))
sub_topic.append(raw_input('Please Enter Sub Topic 02: '))
print 'Making Bullet Points...'

summary_bullets = summary_sents(sub_topic[0], ext_summary)
print "\n"+sub_topic[0]+"\n"
for b in summary_bullets:
    print '* '+b

summary_bullets = summary_sents(sub_topic[1], ext_summary)
print "\n"+sub_topic[1]+"\n"
for b in summary_bullets:
    print '* '+b