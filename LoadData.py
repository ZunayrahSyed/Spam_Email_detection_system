from pathlib import Path
import pandas as pd
import numpy as np


def load_data():
    ham_emails=[]
    emails=Path(r"C:\Users\Dell\OneDrive\Documents\python programs\Spam Email detection system\easy_ham\easy_ham")
    for file in emails.iterdir():
        with open(file,"r") as f:
            ham_emails.append(f.read())
    emails=Path(r"C:\Users\Dell\OneDrive\Documents\python programs\Spam Email detection system\hard_ham\hard_ham")
    for file in emails.iterdir():
        with open(file,"r") as f:
            ham_emails.append(f.read())
    spam_emails=[]
    emails=Path(r"C:\Users\Dell\OneDrive\Documents\python programs\Spam Email detection system\spam_2\spam_2")
    for file in emails.iterdir():
        with open(file,"r",encoding="latin-1") as f:
            spam_emails.append(f.read())
    label_ham=np.zeros(len(ham_emails))
    label_spam=np.ones(len(spam_emails))
    all_emails=ham_emails+spam_emails
    all_labels=label_ham
    all_labels=np.append(all_labels,label_spam)
    ham_dataset=pd.DataFrame(
        {
            'emails':ham_emails,
            'labels':label_ham
        }
    )
    spam_dataset=pd.DataFrame(
        {
            'emails':spam_emails,
            'labels':label_spam
        }
    )
    
    ham_idx=int(len(ham_emails)*0.85)
    spam_idx=int(len(spam_emails)*0.85)
    train=pd.concat([ham_dataset.iloc[0:ham_idx,:],spam_dataset.iloc[0:spam_idx,:]],ignore_index=True)
    test=pd.concat([ham_dataset.iloc[ham_idx:,:],spam_dataset.iloc[spam_idx:,:]],ignore_index=True)
    shuffle_train=np.random.permutation(len(train))
    shuffle_test=np.random.permutation(len(test))
    train=train.iloc[shuffle_train]
    test=test.iloc[shuffle_test]
    perc_ham=((train['labels']==0).sum()/len(train))*100
    perc_spam=((train['labels']==1).sum()/len(train))*100
    print("=======================================================================================================")
    print("===============================   Training Dataset Class Division   ===================================")
    print("Ham: ",perc_ham,"%")
    print("Spam:",perc_spam,"%")
    print("=======================================================================================================")
    return train,test

