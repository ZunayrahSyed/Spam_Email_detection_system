# Spam Email Detection System
This project is built using Python, NumPy, Pandas, and it works on the basis of Logistic Regression.

## Description
The project uses SpamAssassin Public Corpus dataset to train the model using logistic regression to identify spam vs ham emails. It then labels any entered email as spam or ham.

## Dataset
The project uses SpamAssassin Public Corpus from kaggle, which includes the folders easy ham, hard ham, and spam to train the model.
The dataset is not included here, but it is free and publicly available.
Link: https://www.kaggle.com/datasets/beatoa/spamassassin-public-corpus

## Technologies Used
* Python
* NumPy
* Pandas
* Matplotlib

## How it runs
1. Download SpamAssassin Public Corpus from kaggle
2. Add the path of folders `easy_ham`, `hard_ham` , `spam_2`
3. Make sure NumPy, Pandas, and Matplotlib are installed
4. Run the code
5. Input any email to get it labeled as spam or ham

## How it works
This project uses Pandas to load and clean the emails from folders `easy_ham`, `hard_ham` , `spam_2` and then split them into training and test tests using 80/20 ratio. 

It then clean and tokenizes each email. Afterwards, it builds a vocabulary of words whose occurances are in more than 15 emails. It then constructs bag of words using TF-IDF method. 

The model is then trained on training data using learning rate of 1 for 3000 epochs, and evaluated using Confusion Matrix, Accuracy, Precision, Recall, and F1.

The Precision and Recall are compared along different thresholds and the best threshold is selected for the classification of emails after evaluation.
AFterwards, it is ready to classify any email as spam or ham.

## Loss Curve
![Training Loss Curve](losscurve.png)

## Precision vs Recall Curve
![Precision VS Recall Curve for different threshold](PvsRcurve.png)

## What I learned
I learned to extract, clean, and prepare data using Pandas. Moreover, I learned to perform fearure engineering, and TF-IDF method.

Since the project is made using NumPy from sctrach, it gave me a better understanding of the mechanism behind the working of Logistic Regression i.e. how to train data and how to evaluate it using different parameters such as precision and accuracy.
