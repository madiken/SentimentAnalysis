# -*- coding: utf-8 -*- 
from nltk.util import ngrams
import re;
import collections
from prepositions import one_word_prepositions_list, garbage_words_list
from nltk.stem.snowball import RussianStemmer

rs = RussianStemmer(0) 
stop_symbols = [';', '.']
N = 2


     
def removeSlashN(string):

    
    start = 0
    i = string.find(u"\\n")
  
    ss = []
    while i != -1 :
        
       if  (start < len(string)):
            if (len(string[start:i]) > 0 ):
               ss.append(string[start:i])     
            start = i+2  
            i = string.find("\\n", start)
    ss.append(string[start:len(string)])  
    res = ''.join(ss).strip()       
    
    return res

def getFrequenceDict(sentences):
    model_dict = dict()
    
    for i in range(1, N+1) : 
        for s in sentences:
              updateFrequencies(s, i, model_dict)
    return getFrequenciesByModel(model_dict)       

def getModel(sentences):
    model_dict = dict()
    
    for i in range(1, N+1) : 
        for s in sentences:
              updateFrequencies(s, i, model_dict)
    return Model(model_dict, getFrequenciesByModel(model_dict))  
          
  
def train(train_corpus, models, counts):      
    neg_sentences = []
    pos_sentences = []
    
    model_positive = dict()
    model_negative = dict()
    countNeg = 0
    countPos = 0
    
    for t in train_corpus :  
        if (t[1] == 0):
            for i in range(1, N+1) : 
                #sentences = preprocessing(t[0])
                neg_sentences.extend(preprocessing(t[0]))
                countNeg = countNeg + 1
               # for s in sentences:
               #     updateFrequencies(s, i, model_negative)
           # train_corpus_neg.extend(preprocessing(t[0]))
        elif (t[1] == 1):
            for i in range(1, N+1) : 
                pos_sentences.extend(preprocessing(t[0]))
                countPos = countPos + 1
           # train_corpus_pos.extend(preprocessing(t[0]))  
    models.append(getModel(neg_sentences))
    models.append(getModel(pos_sentences))
    
    counts.append(countNeg)
    counts.append(countPos)
    
               
    """ for i in range(1, N+1) :      
        for t in train_corpus_pos :
            updateFrequencies(t, i, model_positive)
            
        for t in train_corpus_neg :
            updateFrequencies(t, i, model_negative)
            
    """
    
    
    
#sentence  
#n - n in n-gram 
#frequencies_to_update dict of dict    
def updateFrequencies(sentence, n, frequencies_to_update): 
   
    ngramz = ngrams(sentence.split(), n)    
   
    
    for ngr in ngramz :    
        ngr = list(ngr)
        
     
        history = ngr[:-1]
        history = (' '.join(history))
        lastWord = ngr[-1:]
        lastWord = (' '.join(lastWord))
          
        
        if (history in frequencies_to_update) :                                      
            if (lastWord in frequencies_to_update[history]) : 
                frequencies_to_update[history][lastWord] = frequencies_to_update[history][lastWord] + 1
            else :
                frequencies_to_update[history][lastWord] = 1                                             
        else :
            frequencies_to_update[history] = dict()
            frequencies_to_update[history][lastWord] = 1
        

      
def preprocessing(t):
    sentences = extractSentences(t)
    res = []
    garbage = re.compile(u'\W+?', re.U)  # [.,:;\'\"()@*_]
    
    for s in sentences:
        tokens = s.split();
        result = []
        
        appendNot = False
        for word in tokens[:] :
            if word == 'не' :
                appendNot = True
                         
            #if (not word in one_word_prepositions_list) :
            #if (not word in garbage_words_list) :       
            else :
                if not appendNot:
                    result.append(normalizeWord(word))
                else :
                    result.append('не' + normalizeWord(word))
                    appendNot = 'False'
                    
        t =' '.join(result)        
        t1 = garbage.sub(' ', t)      
        res.append("<SENTENCE> " + t1 + " </SENTENCE>")  
    return res;


def extractSentences(text):
    text = ';'+text+';'
    sentence_borders = [0];
    sentence_borders.extend([i for i, ltr in enumerate(text) if ltr in stop_symbols])
    sentence_borders.append(len(text) - 1)
    result = []
    start = sentence_borders[0];
    for i in sentence_borders:
        if (text[start: i].strip()!=''):
            result.append(text[start: i].strip())
        start = i + 1   
   
    return result


def normalizeWord(w):      
    #trying not to use pymorphy
    return rs.stem(w).lower()
    
    """ww = re.match(u'[а-яА-ЯёЁ]+', w)
    if ww == None:
        return w.lower()
    else :
        w = ww.group(0)
    l = [x for x in morph.normalize(w.upper())]
    l.sort()
    if (len(l) > 0) :
        return l[0].lower()  
    else :
        return w.lower()       """
             


def getFrequenciesByModel(model_dict):
      result = dict()
      for h in model_dict:
                result[h] = 0
                for l in model_dict[h]:
                    result[h] = result[h] + model_dict[h][l]
      return result
                         
class Model:      
        def __init__(self, model_dict, frequencies): 
            self.structure = model_dict
            self.freq = frequencies
            
            #1 - unigrams, 2- bigrams,...
            self.frequencyRangeMap = {}
        
        def getNoiseBarrier(self, n):
            return len(self.frequencyRangeMap[n].keys())/3    
            
        def getNumberOfNgrams(self, n):
            if n not in self.frequencyRangeMap:
                raise Exception("n is invalid!")
            res = 0
            
            #nb = self.getNoiseBarrier(n)
           # print("nb " , nb)
            #bleft = self.getKthFrequencyFromStart(n, nb)
            #bright = self.getKthFrequencyFromEnd(n, nb)
            
            for k in self.frequencyRangeMap[n].keys():
               # if k <= bleft or k >= bright :
               #     continue
              #  else:
                    res = res + k * self.frequencyRangeMap[n][k]
            
            return res
           
        def setFrequencyVectorForN(self, n):
            self.frequencyRangeMap[n] = self.getFrequencyVectorForN(n) 
            return  self.frequencyRangeMap[n]
            
         
        def getKthFrequencyFromStart(self, n, k):
            if n not in self.frequencyRangeMap:
                raise Exception("n is invalid!")
            
            keys = self.frequencyRangeMap[n].keys()
           # print(len(keys),  " ", k)
            if len(keys) < k or k <= 0:
                raise Exception("k is invalid!")
            
            return keys[k - 1]
        
        def getKthFrequencyFromEnd(self, n, k):
            if n not in self.frequencyRangeMap:
                raise Exception("n is invalid!")
            
            keys = self.frequencyRangeMap[n].keys()
            r = len(keys) - k
            if len(keys) <= r or r < 0:
                raise Exception("k is invalid!")
            
            return keys[r]
                
        def getFrequencyVector(self):
            result = []
            for h in self.structure :
                for l in self.structure[h]:
                    if self.structure[h][l] not in result:
                        result.append(self.structure[h][l]) 
            result.sort(cmp=None, key=None, reverse=False)
            return result            
        
        #i - i-gram i-2 spaces are in history
        def getFrequencyVectorForN(self, i):
            d = dict()
            for h in self.structure :
                if i > 1 and (h.count(u" ") != i - 2 or h ==''):
                    continue
                elif  i == 1 and h != '' :
                    continue
                
                for l in self.structure[h]:
                    if self.structure[h][l] not in d:
                        d[self.structure[h][l]] = 1
                    else :
                        d[self.structure[h][l]] = d[self.structure[h][l]] + 1
            result = collections.OrderedDict(sorted(d.items()))                 
           
            return result     

modelDict = { 'jaja' : {'lo': 3},
              'kaka' : {'mo': 2}, 
              'gjis' : {'mo': 2, 
                        'asa':3}, 
              '' :     {'mo': 2, 
                        'mof': 2, 
                        'fa'  : 2,                        
                       'ololo' : 2, 
                       'frfrddfr' : 3, 
                       'frfrfr' : 4} 
             }
"""
m = Model(modelDict, {})            

print(m.setFrequencyVectorForN(1))
print(m.frequencyRangeMap[1])
print(m.frequencyRangeMap[1].keys())

print(m.setFrequencyVectorForN(2))

print(m.frequencyRangeMap[2])

print(m.getKthFrequencyFromStart(1, 3))

print(m.getKthFrequencyFromEnd(1, 2))
 
print("!" , m.getNumberOfNgrams(1))            
              
                """