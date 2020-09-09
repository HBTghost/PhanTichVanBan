# DON'T CHANGE this part: import libraries
import numpy as np
import scipy
import json
from nltk.stem import PorterStemmer 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
import re
import itertools

# DON'T CHANGE this part: read data path
# train_set_path, valid_set_path, random_number = input().split()
train_set_path, valid_set_path, random_number = '/home/brian/Study/2020/Toan TK & UD/5/PhanTichVanBan/train.json', '/home/brian/Study/2020/Toan TK & UD/5/PhanTichVanBan/valid.json', '0'

# Class Rate init with input
class Rate:
  def __init__(self, train_set_path, valid_set_path, random_number):
    self.train_set_path = train_set_path
    self.valid_set_path = valid_set_path
    self.random_number = int(random_number)
    self.valid_at_random_number = []
    self.train_vocab = []

    self.train_A, self.train_B = self.load_data(is_training=True)
    self.valid_A, self.valid_B = self.load_data(is_training=False)

  def accuracy(self):
    x_hat = np.linalg.pinv(self.train_A) @ self.train_B
    vB = np.argmax(scipy.special.softmax(self.valid_B, axis=1), axis=1) + 1
    pB = np.argmax(scipy.special.softmax(self.valid_A @ x_hat, axis=1), axis=1) + 1
    return np.sum(vB == pB) / vB.shape[0]

  def preprocess(self, text, is_training):
    # Converting text to lowercase
    text = text.lower()
    # converting number
    text = re.sub(r'[0-9]+', 'num', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Removeing stopword
    tokens = [w for w in tokens if not w in stopwords.words()]
    # Stemming
    ps = PorterStemmer()
    stemming = [ps.stem(w) for w in tokens]
    # Convert to unk if not learn
    if not is_training:
      stemming = [w if w in self.train_vocab else 'unk' for w in stemming]
    # Return text preprocessed
    return stemming

  def embedding(self, docs, is_training):
    unique = set()
    for doc in docs:
      for word in doc:
        unique.add(word)
    unique = list(unique)
    if is_training and 'unk' not in unique:
      unique.append('unk')
      self.train_vocab = unique[:]
    if not is_training:
      unique = self.train_vocab

    return np.array([[doc.count(word) for word in unique] for doc in docs])

  def standardized(self, lables):
    res = []
    for label in lables:
      tmp = [0, 0, 0, 0, 0]
      tmp[int(label)-1] = 1
      res.append(tmp)
    return np.array(res)


  def load_data(self, is_training=False):
    path = self.train_set_path if is_training else self.valid_set_path
    with open(path, 'r') as j:
        json_data = json.load(j)[:100]
        preprocessed_data = [self.preprocess(x['reviewText'], is_training) for x in json_data]
        if is_training:
          self.valid_at_random_number = preprocessed_data[self.random_number]
        embedding_data = self.embedding(preprocessed_data, is_training)
        standardized_label = self.standardized([x['overall'] for x in json_data])
        return np.insert(embedding_data, 0, 1, axis=1), standardized_label


rate = Rate(train_set_path, valid_set_path, random_number)
print (rate.valid_at_random_number)
print ('M2 - {}'.format(rate.accuracy()))