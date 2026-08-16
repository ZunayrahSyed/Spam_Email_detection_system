import string
from bs4 import BeautifulSoup


stop_words=['of','the','is','am','in','to','or','from','it','an','a','at','on','as','are','for','and','be','but','so','if',
                'was','were','has','have','can','could','did','do','does','by','with','yet','must','you','i','since','becuase','hence']
def tokenization(email):
    prev=email[0]
    now=email[0]
    count=0
    for i in email:
        count+=1
        prev=now
        now=i
        if prev=='\n' and now=='\n':
            break
    '''useful=""
    for i in range(count,len(email)):
        useful+=email[i]'''
    useful = email[count:]
    useful=useful.lower()
    if useful.find('<html>')!=-1:
        obj=BeautifulSoup(useful,'html.parser')
        useful=obj.get_text()
    while useful.find('http')!=-1:
        start=useful.find('http')
        if start!=-1:
            end=start
            while end+1<len(useful) and useful[end+1]!=' ' and useful[end+1]!='\n':
                end+=1
            end+=1
            useful=useful.replace(useful[start:end],'URLTOKEN')
    useful=useful.replace('\n'," ")
    for i in useful:
        if i in string.punctuation:
            useful=useful.replace(i,'')
    for i in stop_words:
        useful=useful.replace(' '+i+' ',' ')
    while useful.find('  ')!=-1:
        useful=useful.replace('  ',' ')
    tokenized=useful.split(' ')
    for i in range(0,len(tokenized)):
        values=[]
        for ch in tokenized[i]:
            values.append(ch.isdigit())
        if any(values):
            tokenized[i]='NUMTOKEN'
    return tokenized


def tokenization_predictor(useful):
    useful=useful.lower()
    if useful.find('<html>')!=-1:
        obj=BeautifulSoup(useful,'html.parser')
        useful=obj.get_text()
    while useful.find('http')!=-1:
        start=useful.find('http')
        if start!=-1:
            end=start
            while end+1<len(useful) and useful[end+1]!=' ' and useful[end+1]!='\n':
                end+=1
            end+=1
            useful=useful.replace(useful[start:end],'URLTOKEN')
    useful=useful.replace('\n'," ")
    for i in useful:
        if i in string.punctuation:
            useful=useful.replace(i,'')
    for i in stop_words:
        useful=useful.replace(' '+i+' ',' ')
    while useful.find('  ')!=-1:
        useful=useful.replace('  ',' ')
    tokenized=useful.split(' ')
    for i in range(0,len(tokenized)):
        values=[]
        for ch in tokenized[i]:
            values.append(ch.isdigit())
        if any(values):
            tokenized[i]='NUMTOKEN'
    return tokenized


