# -*- coding: utf-8 -*- 
from BeautifulSoup import BeautifulSoup as bs
from test import removeSlashN

def deserialize(filename):
    
    rvpairs = list()
    
    f = open(filename)
    text = unicode(f.read(), "utf-8")
    soup = bs(text)
    f.close()
    
    cnt = 0
    for row in soup('row'):
        label = int(float(row('value')[0].text.strip()))
        if label > 5: label = 1 
        else: label = 0
        
        review = removeSlashN(row('value')[4].text.decode('utf8'))
        print("------------")
        print(review)
        print(label)
        
        rvpairs.append([review,label])
        cnt+=1

    return rvpairs

#deserialize("../big_test.html")
