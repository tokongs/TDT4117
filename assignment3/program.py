import random
import codecs
import string 
from nltk.stem.porter import PorterStemmer
import gensim
import urllib


# Download list of words to exclude
req = urllib.request.Request("https://www.textfixer.com/tutorials/common-english-words.txt", headers={'User-Agent' : "Python"}) 
f = urllib.request.urlopen( req )
exclusion_list = f.read().decode("utf-8").split(",")

# Process paragraphs to make list of words for each paragraph
random.seed(123)
f = codecs.open("pg3300.txt", "r", "utf-8")
stemmer = PorterStemmer()

paragraphs = [p for p in f.read().split("\n\n")]
processed = [p for p in paragraphs if p != "" and "Gutenberg" not in p]
processed = [p.translate(str.maketrans("", "", string.punctuation + "\n\r\t")) for p in processed]
processed = [p.lower().split() for p in processed]
processed = [[ stemmer.stem(w) for w in p ] for p in processed]

dictionary = gensim.corpora.Dictionary(processed)

exclusion_list = [dictionary.token2id[word] for word in exclusion_list if word in dictionary.token2id]
dictionary.filter_tokens(exclusion_list)

bows = [dictionary.doc2bow(p) for p in processed]
