from nltk import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from preprocess import pre_process


def summary_sents(sub_topic, sents):
    sents = sent_tokenize(' '.join(sents))
    sents.append(sub_topic)
    
    sent_vectorizer = TfidfVectorizer(decode_error='replace',min_df=1, stop_words='english',
                                 use_idf=True, tokenizer=pre_process, ngram_range=(1,3))

    sent_tfidf_matrix = sent_vectorizer.fit_transform(sents)
    #subtopic_tfidf = sent_tfidf_matrix.transform([sub_topic])
    sub_topic_similarity = cosine_similarity(sent_tfidf_matrix)
    top10_sents = sub_topic_similarity[-1][:-1].argsort()[:-11:-1]
    final_sents=[]
    for i in top10_sents:
        final_sents.append(sents[i])
    return final_sents