from PrepareData import tokenization
from FeatureEngineering import bag_of_words
from training import sigmoid 
import numpy as np
import matplotlib.pyplot as plt


def evaluation(selected,test,w,b,idf):
    threshold=0.48
    test['vocabulary']=test['emails'].apply(tokenization)
    x,_=bag_of_words(selected,test,idf)
    z=x@w+b
    y_pred=sigmoid(z)
    tp=np.sum(((y_pred>threshold).flatten()) & (test['labels']==1))
    tn=np.sum(((y_pred<=threshold).flatten()) & (test['labels']==0))
    fp=np.sum(((y_pred>threshold).flatten()) & (test['labels']==0))
    fn=np.sum(((y_pred<=threshold).flatten()) & (test['labels']==1))
    print("=======================================================================================================")
    print("======================================   Confusion Matrix   ===========================================")
    print("True Positive(spam emails): ",tp)
    print("True Negative(Ham emails): ",tn)
    print("False Positive(Actually Ham but classified as spam)(Precision): ",fp)
    print("false Negative(Actually Spam but classified as Ham): ",fn)
    correct=((tp+tn)/len(test))*100
    print("=======================================================================================================")
    print("==========================================   Accuracy   ===============================================")
    print("Total Emails: ",len(test),"      Correctly Predicted: ",tp+tn)
    print("Accuracy: ",correct,"%")
    total_spam=np.sum(test['labels']==1)
    precision=(tp/(tp+fp))*100
    print("=======================================================================================================")
    print("==========================================   Precision   ===============================================")
    print("Classified as Spam:",tp+fp,"     Actually spam: ",tp)
    print("Precision: ",precision,"%")
    recall=(tp/total_spam)*100
    print("=======================================================================================================")
    print("==========================================   Recall   ===============================================")
    print("Total Spam: ",total_spam,'       Spam caught: ',tp)
    print("Recall: ",recall,"%")
    F1=(2*precision*recall)/(precision+recall)
    print("=======================================================================================================")
    print("=============================================   F1   ==================================================")
    print("F1: ",F1)
    thresholds=[0.35,0.39,0.42,0.48,0.51,0.56,0.62,0.68,0.71,0.76,0.8]
    precisions=np.array([])
    recalls=np.array([])
    for threshold in thresholds:
        tp=np.sum(((y_pred>threshold).flatten()) & (test['labels']==1))
        tn=np.sum(((y_pred<=threshold).flatten()) & (test['labels']==0))
        fp=np.sum(((y_pred>threshold).flatten()) & (test['labels']==0))
        fn=np.sum(((y_pred<=threshold).flatten()) & (test['labels']==1))
        precisions=np.append(precisions,(tp/(tp+fp))*100)
        recalls=np.append(recalls,(tp/total_spam)*100)
    plot_graph(thresholds,precisions,recalls)
    difference=abs(precisions-recalls)
    return thresholds[np.argmin(difference)]
    
    

def plot_graph(thresholds,precisions,recalls):
    plt.figure(figsize=(8,6))
    plt.plot(thresholds,precisions,marker='.',color='purple',label='Precisions')
    plt.plot(thresholds,recalls,marker='.',color='brown',label='Recalls')
    plt.legend()
    plt.title('Precision & Recall for different Thresholds',color="Black",fontsize=20)
    plt.xlabel('Threshold',fontsize=15)
    plt.grid(linestyle="--",color="lavender")
    plt.xticks(thresholds)
    plt.yticks(np.arange(10,101,5))
    plt.savefig('Precision & recall vs Threshold.png',dpi=300)
    plt.show()


