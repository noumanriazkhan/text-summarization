
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import networkx as nx

def summarize(text):
    
    sentences_token = sent_tokenize(text)
    
    #Feature Extraction
    vectorizer = CountVectorizer(min_df=1,decode_error='replace')
    sent_bow = vectorizer.fit_transform(sentences_token)
    transformer = TfidfTransformer(norm='l2', smooth_idf=True, use_idf=True)
    sent_tfidf = transformer.fit_transform(sent_bow)
    
    similarity_graph = sent_tfidf * sent_tfidf.T
    
    nx_graph = nx.from_scipy_sparse_matrix(similarity_graph)
    scores = nx.pagerank(nx_graph)
    text_rank_graph = sorted(((scores[i],s) for i,s in enumerate(sentences_token)), reverse=True)
    number_of_sents = int(0.4*len(text_rank_graph))
    del text_rank_graph[number_of_sents:]
    summary = ' '.join(word for _,word in text_rank_graph)
    
    return summary
    