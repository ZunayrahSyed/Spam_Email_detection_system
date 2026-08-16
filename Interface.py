from LoadData import load_data
from PrepareData import tokenization,tokenization_predictor
from FeatureEngineering import build_vocabulary,bag_of_words,bag_of_words_predictor
from training import training
from Evaluation import evaluation
import numpy as np

def predictor(email,selected,threshold,w,b,selected_index,idf):
    tokenized=tokenization_predictor(email)
    x=bag_of_words_predictor(selected,selected_index,tokenized,idf)
    z=x@w+b
    if z>=0:
        y_pred=1/(1+np.exp(-z))
    else:
        y_pred=np.exp(z)/(1+np.exp(z))

    if y_pred>threshold:
        print("The email is Spam")
        print("Confidence: ",y_pred*100,"%")
    else:
        print("The email is Ham")
        print("Confidence: ",(1-y_pred)*100,"%")

learning_rate=1
epochs=3000
train,test=load_data()
train['vocabulary']=train['emails'].apply(tokenization)
selected,idf=build_vocabulary(train)
matrix,selected_index=bag_of_words(selected,train,idf)
w,b=training(matrix,train['labels'],learning_rate,epochs)
threshold=evaluation(selected,test,w,b,idf)
print("=======================================================================================================")
print("==========================================   Predictor   ==============================================")
email=input("Enter the email you wanna classify as spam or ham: ")
predictor(email,selected,threshold,w,b,selected_index,idf)