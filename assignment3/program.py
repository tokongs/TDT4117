import random
import codecs
import string 
from nltk.probability import FreqDist
from nltk.stem.porter import PorterStemmer
import gensim
import urllib

random.seed(123)

#Remove punctuation, tokenize and stem query
def preprocessing(query):
    return [ stemmer.stem(word) for word in query.translate(str.maketrans("", "", string.punctuation + "\n\r\t")).lower().split()]

# Only keep the n lines in the paragraph
def truncate_lines(paragraph, number_of_lines):
    lines = paragraph.splitlines()
    if len(lines) <= number_of_lines:
        return paragraph
    return "".join(lines[:number_of_lines])


# Download list of words to exclude
req = urllib.request.Request("https://www.textfixer.com/tutorials/common-english-words.txt", headers={'User-Agent' : "Python"}) 
f = urllib.request.urlopen( req )
exclusion_list = f.read().decode("utf-8").split(",")

# Load and process paragraphs to make list of words for each paragraph

f = codecs.open("pg3300.txt", "r", "utf-8")
stemmer = PorterStemmer()


paragraphs = [p for p in f.read().split("\n\n")]
paragraphs = [p for p in paragraphs if p != "" and "Gutenberg" not in p]
processed = [p.translate(str.maketrans("", "", string.punctuation + "\n\r\t")) for p in paragraphs]
processed = [p.lower().split() for p in processed]
processed = [[ stemmer.stem(w) for w in p ] for p in processed]

fdist = FreqDist([word for paragraph in processed for word in paragraph])
fdist = FreqDist(dict(fdist.most_common()[:15]))

# Create dictionary
dictionary = gensim.corpora.Dictionary(processed)
exclusion_list = [dictionary.token2id[word] for word in exclusion_list if word in dictionary.token2id]
dictionary.filter_tokens(exclusion_list)

# Create Bags of words
bows = [dictionary.doc2bow(p) for p in processed]

# Create TF-IDF model
tfidf_model = gensim.models.TfidfModel(bows)
tfidf_corpus = [tfidf_model[p] for p in bows]

tfidf_index = gensim.similarities.MatrixSimilarity(tfidf_corpus)

# Create LSI model
lsi_model = gensim.models.LsiModel(tfidf_corpus, id2word=dictionary, num_topics=100)
lsi_corpus = [lsi_model[p] for p in bows]

lsi_index = gensim.similarities.MatrixSimilarity(lsi_corpus)

# Print 3 first lsi topics
print("3 first LSI topics:")
for topic in lsi_model.show_topics()[:3]:
    print(topic)
print("\n")

# Create query
query = preprocessing("What is the function of money?")
query_bow = dictionary.doc2bow(query)
tfidf_query = tfidf_model[query_bow]

# Print tf-idf weights for query
print("TF-IDF weights for query:")
for token in tfidf_query:
    print(f"{dictionary.get(token[0], token[1])}: {token[1]}")
print("\n")


# Find three most relevant paragraphs
doc2similarity = enumerate(tfidf_index[tfidf_query])
docs = sorted(doc2similarity, key=lambda kv: -kv[1])[:3]
print
print("Ttop 3 relevant pararaphs acording to TF-IDF model:")
for doc in docs:
    print(f"[Paragraph {doc[0]}, Similarity {doc[1]}]")
    print(truncate_lines(paragraphs[doc[0]], 5))
    print("\n")

# Create lsi query and find top 3 topics
lsi_query = lsi_model[tfidf_query] 
topics = sorted(lsi_query, key=lambda kv: -abs(kv[1]))[:3]

# Print top 3 topics
print("Top 3 LSI topics")
for topic in topics:
    print(f"[Topic {topic[0]}")
    print(lsi_model.show_topics()[topic[0]])
    print("\n")


# Find top 3 most relevant documents
doc2similarity = enumerate(lsi_index[lsi_query])
docs = sorted(doc2similarity, key=lambda kv: -kv[1])[:3]

# Print top 3 most relevant documents
print("Ttop 3 relevant pararaphs acording to LSI model:")
for doc in docs:
    print(f"[Paragraph {doc[0]}, Similarity {doc[1]}]")
    print(truncate_lines(paragraphs[doc[0]], 5))
    print("\n")

fdist.plot(cumulative=False)
