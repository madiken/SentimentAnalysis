# -*- coding: utf-8 -*-
from deserializer import deserialize
from test import train, preprocessing, N, getModel
from nltk.util import ngrams
import math

#ngr_list n-gram in list 
def getHistory(ngr_list):
    return (' '.join(ngr_list[:-1]))

def getLastWord(ngr_list):
    return (' '.join(ngr_list[-1:]))



def getWordsList(string):
    result = []
    ngrs =  ngrams(string.split(), 1)
    for n in ngrs :
        result.extend(list(n))
    return result



#    classModel : model_positive or model_negative
#    features :  pairs (ngrams + frequncy dict by histories)
def getConditionalClassProbabilityLog(classModel_obj, obj):
    #print("getConditionalClassProbability------------")
    
    #resultLog = 0
    result = 1
    for f in obj.ngrams :
        bo = backOffProbability(classModel_obj, f, obj.model)
              
        if bo != 0: #remove noise 
           # resultLog = resultLog + math.log(bo)                   
           result = result * bo
    return result       


def printBigramModel(model, filename):
   fo = open(filename, "wb")
   for history in model :
    fo.write("history " )
    
    fo.write(";"+history+";\n")
    fo.write(str(type(history)))
    
    for lastWord in model[history].keys() :       
        
        fo.write("\n             lastWord "  + lastWord + " " + str(type(lastWord)) +" " + str(model[history][lastWord]) +"\n")
       



def starFrequency(classModel_obj, history, lastWord, obj_model) :
    cwh = 0 
    if (history in classModel_obj.structure) :
        
        if (lastWord in classModel_obj.structure[history]):
            cwh = classModel_obj.structure[history][lastWord]    
        
        if cwh == 0 :
            return 0     
        
        elif (history in obj_model.structure) :
            
            if (lastWord in obj_model.structure[history]):
                cwh = cwh + obj_model.structure[history][lastWord]
                
    return cwh

def starHistoryFrequency(classModel_obj, history, obj_model) :
    ch = 0 
    
    if (history in classModel_obj.structure):
            ch = ch + classModel_obj.freq[history]
        
    if (history in obj_model.structure):
            ch = ch + obj_model.freq[history] 
    
        
    """if ('' in classModel_obj.structure) :
        if (history in classModel_obj.structure['']):
            ch = classModel_obj.structure[''][history]
        
       
    if ('' in obj_model.structure) :
        if (history in obj_model.structure['']):
            ch = ch + obj_model.structure[''][history]"""
                
    return ch

#p* = C(w|h)/C(h)
def starProbability(classModel_obj, ngr_list, obj_model) :
    
    classModel = classModel_obj.structure
    #print("starProbability-------------------")
    #print(' '.join(ngr_list))
   
    history = getHistory(ngr_list)
    lastWord = getLastWord(ngr_list)    
    cwh = 0
    
    if (history in classModel_obj.structure) :
        
        if (lastWord in classModel[history]):
            cwh = classModel[history][lastWord]    
        history_freq = classModel_obj.freq[history]     
                        
        
        if cwh == 0 :
            return 0     
       ##""" 
        elif (history in obj_model.structure) :
            
            if (lastWord in obj_model.structure[history]):
                cwh = cwh + obj_model.structure[history][lastWord]
                
                
        history_freq = starHistoryFrequency(classModel_obj, history, obj_model) 
                  
    if (cwh != 0) :
        result = (cwh)*1.0/(history_freq)     
    else :
        result = 0        
        
    return result

    
def smallFakeProbability(classModel_obj, obj_model) : 
    
    freq = 0
    
    if ('' in classModel_obj.structure) :
        freq = freq + classModel_obj.freq['']
    if freq == 0 :
        return 0
    else :
       """ if '' in obj_model.structure :
            freq = freq + obj_model.freq['']"""
       return  1.0/(freq)/100         
 
 
        
def backOffProbability(classModel_obj, ngr_list, obj_model) :
    
    #print("backoff probability of")
    #print(' '.join(ngr_list))
    

    if (len(ngr_list) == 0) :
        return smallFakeProbability(classModel_obj, obj_model)
       
       
    prob = starProbability(classModel_obj, ngr_list, obj_model)
    #print("star prob ", prob)  
    
    if (prob != 0) :
        return prob
    else :
        classModel = classModel_obj.structure     

        alpha = 0
        history =  getHistory(ngr_list)
        shortHistory = getHistory(ngr_list[1:])
        
        #delete noise
        if not filterOutNgramThroughModel(classModel_obj, len(ngr_list), ngr_list):
            return 0
        
        if ('' == history):
           return smallFakeProbability(classModel_obj, obj_model)
        
        lastWord = getLastWord(ngr_list)
        sigmaPStarFullMinu = 0 
        sigmaPStarShortMinu = 0
        if history in classModel :
            #print("history in classModel")
            for l in classModel[history] :
               
                #minu = starProbability(classModel_obj, getWordsList(history + " " + l), obj_model)
                sigmaPStarFullMinu = sigmaPStarFullMinu + starFrequency(classModel_obj, history, l, obj_model)
                sigmaPStarShortMinu = sigmaPStarShortMinu + starFrequency(classModel_obj, shortHistory, l, obj_model)
            """print("minu ", minu)
                print("sigmaPStarFull ", sigmaPStarFull)
                
                sigmaPStarFull = sigmaPStarFull - minu
                print("sigmaPStarFull ", sigmaPStarFull)"""
                
                ##print("minu ", minu)
                ##print(sigmaPStarFull)
        history_freq = starHistoryFrequency(classModel_obj, history, obj_model)        
        shortHistory_freq = starHistoryFrequency(classModel_obj, shortHistory, obj_model)        
        
        sigmaPStarFull = 1.0
        sigmaPStarShort = 1.0
        if history_freq != 0 :
            sigmaPStarFull = sigmaPStarFull - sigmaPStarFullMinu*1.0/history_freq
        if shortHistory_freq!=0 :
            sigmaPStarShort = sigmaPStarShort - sigmaPStarShortMinu*1.0/shortHistory_freq
           
        
        """
        print("history ")
        print(history)
        print("short history ")
        print(shortHistory)
        print("lastword ")
        print(lastWord)
             
        print("sigmaPStarFull" , sigmaPStarFull)
        print("sigmaPStarShort" , sigmaPStarShort)"""
        if (sigmaPStarShort == 0 ) and (sigmaPStarFull == 0):
            alpha = 1
        else :    
            alpha = sigmaPStarFull* 1.0/sigmaPStarShort   
       
        res = alpha*backOffProbability(classModel_obj, ngr_list[1:], obj_model)
        return res
    
def filterOutNgramThroughModel(classModel_obj, n, ngram_list):
        h = getHistory(ngram_list)
        l = getLastWord(ngram_list)
        
        #get backoff probability
        if h not in classModel_obj.structure :
            return True
        elif l not in  classModel_obj.structure[h]:
            return True
        
        if n not in classModel_obj.frequencyRangeMap :
            raise Exception("n is invalid!")
        #magic k
        k = classModel_obj.getNoiseBarrier(n) 
        kleft = len(classModel_obj.frequencyRangeMap[n].keys()) * 0.3
        kright = len(classModel_obj.frequencyRangeMap[n].keys()) * 0.4
          
        try :
            if classModel_obj.structure[h][l] <= classModel_obj.getKthFrequencyFromStart(n, kleft) or classModel_obj.structure[h][l] >= classModel_obj.getKthFrequencyFromStart(n, kright):
                return False
        except :
            pass    
        
        
        return True
    
class SentimentAnalyzer:
    classifier = None    

                   
    # constructor (optional)
    def __init__(self, path_to_training_corpus):
        train_pairs = deserialize(path_to_training_corpus)
        self.train(train_pairs)
        
        
    def filterOutNgram(self, n, ngram_list):     
        if filterOutNgramThroughModel(self.classifier.model_positive, n, ngram_list) or  filterOutNgramThroughModel(self.classifier.model_negative, n, ngram_list):
            return True
        return False 
    # convert texts to feature vectors (optional)
    def featureExtractor(self, texts):
        ngrams_list = []
        for t in texts :            
            sentences = preprocessing(t)                
            text_ngrs = []    
            for s in sentences :    
                            
                ngrs = ngrams(s.split(), N)
                n = N
                while len(ngrs)==0 and n > 1: 
                    n = n - 1
                    ngrs = ngrams(s.split(), n) 
                for ngr in ngrs :   
                    nn = list(ngr)
                    if self.filterOutNgram( n, nn) :
                        text_ngrs.append(nn)    
                   # else :
                   #     print("filter! ", ' '.join(nn))    
           
            #print(len(text_ngrs))         
            ngrams_list.append( ObjectToClassify(text_ngrs, getModel(sentences) ) )
            
        
        return ngrams_list  
            
          
        
    # trainer of classifier (optional)
    def train(self, training_corpus):
        models = [] # 0 -neg  , 1-pos
        counts = []
        train(training_corpus, models, counts)
        

            
        self.classifier = Classifier(models[1], models[0])
        self.classifier.positiveClassProb = counts[1] * 1.0 / (counts[1] + counts[0])
        self.classifier.negativeClassProb = counts[0] * 1.0/ (counts[1] + counts[0])
        
        printBigramModel(self.classifier.model_positive.structure, "pos.txt")
        printBigramModel(self.classifier.model_negative.structure, "neg.txt")
        
       
        
        
    # returns sentiment score of input text (mandatory)
    def getClasses(self, texts):
       
        feature_vectors = self.featureExtractor(texts)
        
        result = []
        for f in feature_vectors : 
            result.append(self.classifier.predict(f)) 
            
        return result
    
 
                 
class Classifier:
    
    
    def __init__(self, model_pos, model_neg):
        self.model_positive = model_pos
        self.model_negative = model_neg  
        
        for i in range(1, N+1):
            self.model_positive.setFrequencyVectorForN(i)
       
        for i in range(1, N+1):
            self.model_negative.setFrequencyVectorForN(i)
            
        """ posNum = self.model_positive.getNumberOfNgrams(N)
        negNum = self.model_negative.getNumberOfNgrams(N)
        
        self.positiveClassProb = posNum * 1.0 /(posNum + negNum)
        self.negativeClassProb = negNum * 1.0 /(posNum + negNum)"""
        
        
    
    def predict(self, obj):
        #print("positiveClassProb ",  self.positiveClassProb)
        #print("negativeClassProb " , self.negativeClassProb)
        
        posProbLog = self.positiveClassProb * getConditionalClassProbabilityLog(self.model_positive, obj)
        
        negProbLog = self.negativeClassProb * getConditionalClassProbabilityLog(self.model_negative, obj)
        
        #print("posProbLog " , posProbLog)
        #print("negProblog " , negProbLog)
        
        if posProbLog >= negProbLog :
            result = 1
        else : 
            result = 0
        return result    

class ObjectToClassify :
    def __init__(self, ngr_list, model):
        self.ngrams = ngr_list
        self.model = model
        
    
                
            
             
