import random
import codecs
import string 

random.seed(123)
f = codecs.open("pg3300.txt", "r", "utf-8")

paragraphs = [p for p in f.read().split("\n\n")]
paragraphs = [p for p in paragraphs if p != "" and "Gutenberg" not in p]
paragraphs = [p.translate(str.maketrans("", "", string.punctuation + "\n\r\t")) for p in paragraphs]
paragraphs = [p.lower().split() for p in paragraphs]