# The name of this file is supposed to be a pun. Classify, classi.py...
import sqlite3 as sqlite
import re
import math

def getwords(doc):
  splitter=re.compile('\W+')  # different than book
  #print (doc)
  # Split the words by non-alpha characters
  words=[s.lower() for s in splitter.split(doc) 
          if len(s)>2 and len(s)<20]
  
  # Return the unique set of words only
  uniq_words = dict([(w,1) for w in words])

  return uniq_words

class basic_classifier:

  def __init__(self,getfeatures,filename=None):
    # Counts of feature/category combinations
    self.fc={}
    # Counts of documents in each category
    self.cc={}
    self.getfeatures=getfeatures
    
  # Increase the count of a feature/category pair  
  def incf(self,f,cat):
    self.fc.setdefault(f, {})
    self.fc[f].setdefault(cat, 0)
    self.fc[f][cat]+=1
  
  # Increase the count of a category  
  def incc(self,cat):
    self.cc.setdefault(cat, 0)
    self.cc[cat]+=1  

  # The number of times a feature has appeared in a category
  def fcount(self,f,cat):
    if f in self.fc and cat in self.fc[f]:
      return float(self.fc[f][cat])
    return 0.0

  # The number of items in a category
  def catcount(self,cat):
    if cat in self.cc:
        return float(self.cc[cat])
    return 0

  # The total number of items
  def totalcount(self):
    return sum(self.cc.values())

  # The list of all categories
  def categories(self):
    return self.cc.keys()

  def train(self,item,cat):
    features=self.getfeatures(item)
    # Increment the count for every feature with this category
    for f in features:
      self.incf(f,cat)

    # Increment the count for this category
    self.incc(cat)

  def fprob(self,f,cat):
    if self.catcount(cat)==0: return 0

    # The total number of times this feature appeared in this 
    # category divided by the total number of items in this category
    return self.fcount(f,cat)/self.catcount(cat)

  def weightedprob(self,f,cat,prf,weight=1.0,ap=0.5):
    # Calculate current probability
    basicprob=prf(f,cat)

    # Count the number of times this feature has appeared in
    # all categories
    totals=sum([self.fcount(f,c) for c in self.categories()])

    # Calculate the weighted average
    bp=((weight*ap)+(totals*basicprob))/(weight+totals)
    return bp

def sampletrain(cl):
  cl.train('Nobody owns the water.','good')
  cl.train('the quick rabbit jumps fences','good')
  cl.train('buy pharmaceuticals now','bad')
  cl.train('make quick money at the online casino','bad')
  cl.train('the quick brown fox jumps','good')

def sampletest_basic(cl):
    sampletrain(cl)
    print("")
    print("Total items:", cl.totalcount())
    print("Categories:", cl.categories())
    for cat in cl.categories():
        print(cat, cl.catcount(cat))
    # Example 1
    print("Example 1:")
    print(cl.fcount('quick', 'good'))
    print(cl.fcount('quick', 'bad'))
    # Example 2
    print("Example 2:")
    print(cl.fprob('quick', 'good'))
    # Example 3
    print("Example 3:")
    print(cl.weightedprob('money', 'bad', cl.fprob))
    print(cl.train("This money is bad.", "bad"))
    print(cl.weightedprob('money', 'bad', cl.fprob))
    # Example 4
    print("Example 4:")
    print(cl.fprob('money', 'good'))
    print(cl.weightedprob('money', 'good', cl.fprob))
    # Example 5
    print("Example 5:")
    print(cl.weightedprob('money', 'good', cl.fprob))
    print(cl.weightedprob('money', 'good', cl.fprob))

# To use this with the basic classifier (and to change it back later), make the following changes:
# class naivebayes(classifier) -> class naivebayes(basic_classifier)
# classifier.__init__(self,getfeatures) -> basic_classifier.__init__(self,getfeatures)

class naivebayes(basic_classifier):   # change for basic_classifier

  def __init__(self,getfeatures):   
    basic_classifier.__init__(self,getfeatures)  # change for basic_classifier
    self.thresholds={}
  
  def docprob(self,item,cat):
    features=self.getfeatures(item)   

    # Multiply the probabilities of all the features together
    p=1
    for f in features: p*=self.weightedprob(f,cat,self.fprob)
    return p

  def prob(self,item,cat):
    catprob=self.catcount(cat)/self.totalcount()
    docprob=self.docprob(item,cat)
    return docprob*catprob
  
  def setthreshold(self,cat,t):
    self.thresholds[cat]=t
    
  def getthreshold(self,cat):
    if cat not in self.thresholds: return 1.0
    return self.thresholds[cat]
  
  def classify(self,item,default=None):
    probs={}
    # Find the category with the highest probability
    max=0.0
    for cat in self.categories():
      probs[cat]=self.prob(item,cat)
      if probs[cat]>max: 
        max=probs[cat]
        best=cat

    # Make sure the probability exceeds threshold*next best
    for cat in probs:
      if cat==best: continue
      if probs[cat]*self.getthreshold(best)>probs[best]: return default
    return best

def sampletest_naive(cl):
    sampletrain(cl)
    print("")
    print("Total items:", cl.totalcount())
    print("Categories:", cl.categories())
    for cat in cl.categories():
        print(cat, cl.catcount(cat))
    # Example 1
    print("Example 1:")
    print(cl.prob('quick rabbit', 'good'))
    print(cl.prob('quick rabbit', 'bad'))
    # Example 2
    print("Example 2:")
    print(cl.classify('quick rabbit', default='unknown'))
    print(cl.classify('quick money', default='unknown'))
    print(cl.setthreshold('bad', 3.0))
    print(cl.classify('quick money', default='unknown'))
    for i in range(10): sampletrain(cl)
    cl.classify('quick money', default='unknown')

def trainAll(cl): # Training Emails
    print("Training:")
    for i in range(1,21):
        # Off-Topic Emails
        filename = "training/off-topic/off-topic-" + str(i) + ".txt"
        print(filename)
        file = open(filename, 'r', encoding='utf8')
        cl.train(file.read(), 'off-topic')
        file.close()
        # On-Topic Training Emails
        filename = "training/on-topic/on-topic-" + str(i) + ".txt"
        print(filename)
        file = open(filename, 'r', encoding='utf8')
        cl.train(file.read(), 'marketing')
        file.close()

def testAll(cl): # Testing Emails
    print("Testing:")
    for i in range(1,6): # Off-Topic Emails
        filename = "testing/off-topic/off-topic-" + str(i) + ".txt"
        file = open(filename, 'r', encoding='utf8')
        print(filename + " - " + cl.classify(file.read(), default='unknown'))
        file.close()
    for i in range(1,6): # On-Topic Training Emails
        filename = "testing/on-topic/on-topic-" + str(i) + ".txt"
        file = open(filename, 'r', encoding='utf8')
        print(filename + " - " + cl.classify(file.read(), default='unknown'))
        file.close()

# Basic_Classifier examples: 
#cl = basic_classifier(getwords)
#sampletest_basic(cl)

# Naive Bayes classifier Examples:
#cl = naivebayes(getwords)
#sampletest_naive(cl)

# My Emails:
cl = naivebayes(getwords)
trainAll(cl)
testAll(cl)