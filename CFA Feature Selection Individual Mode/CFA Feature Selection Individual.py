import math
import pandas as pd
import numpy as np
import scipy as sci
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif

from sklearn import preprocessing

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
from pathlib import Path

def get_acc_indv(clf, X, y,k):

    kf = KFold(n_splits=k, random_state=42, shuffle=False)
    avg_accuracies_list = []
    std_list = []
    #print((X.shape))
    for i in range(X.shape[1]):
        accur_list = []
        acc = 0
        print(i)
        for train_index, test_index in kf.split(X):
            sm = SMOTE(random_state = 42, ratio = 1)
            #print(np.array(X.iloc[:,i].iloc[train_index]))
            X_res, y_res = sm.fit_sample(np.array(X.iloc[:,i].iloc[train_index]).reshape(-1, 1), y.iloc[train_index])
            clf.fit(X_res, y_res)
            y_pred = clf.predict(np.array(X.iloc[:,i].iloc[test_index]).reshape(-1, 1))
            #print(y_pred.shape)
            #print(y.iloc[test_index].shape)
            score = accuracy_score(y_pred, y.iloc[test_index])
            accur_list.append(score)
            #print(X.iloc[:,i-1:i].iloc[test_index])

        avg_accur = sum(accur_list)/k
        print(avg_accur)
        avg_accuracies_list.append(avg_accur)
        sum_of_acc = 0
        for j in range(len(accur_list)):
            sum_of_acc = sum_of_acc +(accur_list[0] - avg_accur)**2
        std = math.sqrt((sum_of_acc)/len(accur_list))
        print(std)
        std_list.append(std)
    return avg_accuracies_list, std_list

def send_to_csv(clf_avg, clf_std, file_name):
    results = pd.DataFrame(dict(clas_avg = clf_avg, clas_std = clf_std))
    results.to_csv(file_name+'.csv')

def main():
    clf1 = LogisticRegression(solver = 'liblinear')
    clf2 = Perceptron(tol=1e-3, random_state=42)
    clf3 = GaussianNB()
    clf4 = SVC(gamma='auto', verbose = 3)
    clf5 = RandomForestClassifier(n_estimators='warn', criterion='gini', max_depth=None,
                              min_samples_split=2, min_samples_leaf=1,
                              min_weight_fraction_leaf=0.0, max_features='auto',
                              max_leaf_nodes=None, min_impurity_decrease=0.0,
                              min_impurity_split=None, bootstrap=True, oob_score=False,
                              n_jobs=None, random_state=None, verbose=0, warm_start=False,
                              class_weight=None)
    clf6 = KNeighborsClassifier(n_neighbors=5)

    path = 'Train_set.csv'
    df = pd.read_csv(path)
    df = df.set_index(['Unnamed: 0'])
    y_train = df['y']
    X_train = df.iloc[:,:89]
    print('This is: clf1')
    clf1_avg, clf1_std = get_acc_indv(clf1, X_train,y_train,10)
    send_to_csv(clf1_avg, clf1_std, 'clf1')
    print('This is: clf2')
    clf2_avg, clf2_std = get_acc_indv(clf2, X_train,y_train,10)
    send_to_csv(clf2_avg, clf2_std, 'clf2')

    print('This is: clf3')
    clf3_avg, clf3_std = get_acc_indv(clf3, X_train,y_train,10)
    send_to_csv(clf3_avg, clf3_std, 'clf3')

    print('This is: clf5')
    clf5_avg, clf5_std = get_acc_indv(clf5, X_train,y_train,10)
    send_to_csv(clf5_avg, clf5_std, 'clf5')

    print('This is: clf6')
    clf6_avg, clf6_std = get_acc_indv(clf6, X_train,y_train,10)
    send_to_csv(clf6_avg, clf6_std, 'clf6')

    print('This is: clf4')
    clf4_avg, clf4_std = get_acc_indv(clf4, X_train,y_train,10)
    send_to_csv(clf4_avg, clf4_std, 'clf4')

    print("FINISHED")

if __name__ == "__main__":
    main()
