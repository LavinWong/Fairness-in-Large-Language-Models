import pickle
from collections import defaultdict, Counter
import numpy as np

STOPWORDS = set(["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"])

def load_dataset(path):
    
    with open(path, "rb") as f:
        
        data = pickle.load(f)
    return data

def load_dictionary(path):
    
    with open(path, "r", encoding = "utf-8") as f:
        
        lines = f.readlines()
        
    k2v, v2k = {}, {}
    for line in lines:
        
        k,v = line.strip().split("\t")
        v = int(v)
        k2v[k] = v
        v2k[v] = k
    
    return k2v, v2k
    
def count_profs_and_gender(data):
    
    counter = defaultdict(Counter)
    for entry in data:
        gender, prof = entry["g"], entry["p"]
        counter[prof][gender] += 1
        
    return counter

def load_data():
    train = load_dataset("data/biasbios/train.pickle")
    dev = load_dataset("data/biasbios/dev.pickle")
    test = load_dataset("data/biasbios/test.pickle")
    counter = count_profs_and_gender(train+dev+test)
    p2i, i2p = load_dictionary("data/biasbios/profession2index.txt")
    g2i, i2g = load_dictionary("data/biasbios/gender2index.txt")

    path = "data/biasbios/"
    x_train = np.load(path + "train_cls.npy")
    x_dev = np.load(path + "dev_cls.npy")
    x_test = np.load(path + "test_cls.npy")
    
    train_min_length = min(len(train), len(x_train))
    train = train[:train_min_length]
    x_train = x_train[:train_min_length]
    
    dev_min_length = min(len(dev), len(x_dev))
    dev = dev[:dev_min_length]
    x_dev = x_dev[:dev_min_length]

    test_min_length = min(len(test), len(x_test))
    test = test[:test_min_length]
    x_test = x_test[:test_min_length]

    y_train = np.array([p2i[entry["p"]] for entry in train])
    y_dev = np.array([p2i[entry["p"]] for entry in dev])
    y_test = np.array([p2i[entry["p"]] for entry in test])

    return p2i, i2p, g2i, i2g, train, dev, test, x_train, y_train, x_dev, y_dev, x_test, y_test