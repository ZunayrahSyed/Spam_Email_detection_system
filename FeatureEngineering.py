import numpy as np


def build_vocabulary(train):
    n=len(train)
    vocab={}
    idf={}
    for i in train['vocabulary']:
        once=set(i)
        for word in once:
            if word not in vocab:
                vocab[word]=1
            else:
                vocab[word]+=1
    selected=[]
    for key,value in vocab.items():
        if value>=15:
            selected.append(key)
            idf[key]=np.log(n/vocab[key])
    return selected,idf

def bag_of_words(selected,train,idf):
    selected_index={}
    for idx in range(len(selected)):
        selected_index[selected[idx]]=idx
    matrix=np.zeros((len(train),len(selected)))
    count=0
    for i in train['vocabulary']:
        for word in i:
            if word in selected_index:
                matrix[count,selected_index[word]]+=1
        count+=1

    count=0
    for list in train['vocabulary']:
        matrix[count,:]/=len(list)
        count+=1

    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            matrix[i,j]*=idf[selected[j]]

    return matrix,selected_index


def bag_of_words_predictor(selected,selected_index,email,idf):
    matrix=np.zeros((1,len(selected)))
    for word in email:
        if word in selected_index:
            matrix[0,selected_index[word]]+=1
    matrix/=len(email)
    for j in range(len(matrix[0])):
        matrix[0,j]*=idf[selected[j]]

    return matrix


