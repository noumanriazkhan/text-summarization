import nltk
from nltk.stem.snowball import SnowballStemmer

stopwords = nltk.corpus.stopwords.words('english')
stemmer = SnowballStemmer("english")

#Defining pre-process function
def pre_process(doc):
    tokens = [word for sent in nltk.sent_tokenize(doc) for word in nltk.word_tokenize(sent)]
    tokens= [w for w in tokens if w not in stopwords]
    tokens = [w.lower() for w in tokens if w.isalpha()]
    stems = [stemmer.stem(t) for t in tokens]
    return stems