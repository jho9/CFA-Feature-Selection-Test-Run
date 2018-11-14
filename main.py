import math
import pandas as pd
import numpy as np
import scipy as sci
#import matplotlib.pyplot as plt

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

def filter_features (X_train, y_train, index, clf):
    sm = SMOTE(random_state = 42, ratio = 1)
    kf = KFold(n_splits=5, random_state=42, shuffle=False)
    list_of_acc = []
    list_of_std = []
    for i in range(len(index)):
        sum_of_acc = 0
        accur_list = []
        print("This is iteration:", i+1, "out of", len(index))
        index_train = X_train.iloc[:,index[0:i+1]]
        #print(index_train)
        for train_index, test_index in kf.split(index_train):
            #print("TRAIN:", train_index, "TEST:", test_index)
            k_X_train, k_X_test = index_train.iloc[train_index], index_train.iloc[test_index]
            k_y_train, k_y_test = y_train.iloc[train_index], y_train.iloc[test_index]
            k_X_train, k_y_train = sm.fit_sample(k_X_train, k_y_train)
        
            clf.fit(k_X_train,k_y_train)
            k_pred_y = clf.predict(k_X_test)
            accur_list.append(accuracy_score(k_pred_y, k_y_test))
        avg_accur = sum(accur_list)/len(accur_list)
        for i in range(len(accur_list)):
            sum_of_acc = sum_of_acc +(accur_list[0] - avg_accur)**2
        std = math.sqrt((sum_of_acc)/len(accur_list))
        #print("The average accuracy is: ",avg_accur, "and the std is: ", std)
        list_of_acc.append(avg_accur)
        list_of_std.append(std)
        df_acc = pd.DataFrame(list_of_acc).T
        df_std = pd.DataFrame(list_of_std).T
    return df_acc, df_std

def get_avg_feat_importance(clf, X_train, y_train):
    sm = SMOTE(random_state = 42, ratio = 1)
    kf = KFold(n_splits=5, random_state=42, shuffle=False)
    accur_list = []
    feat_import =[]
    sum_of_acc = 0
    for train_index, test_index in kf.split(X_train):
        #print("TRAIN:", train_index, "TEST:", test_index)
        k_X_train, k_X_test = X_train.iloc[train_index], X_train.iloc[test_index]
        k_y_train, k_y_test = y_train.iloc[train_index], y_train.iloc[test_index]
        k_X_train, k_y_train = sm.fit_sample(k_X_train, k_y_train)
        clf.fit(k_X_train,k_y_train)
        k_pred_y = clf.predict(k_X_test)
        accur_list.append(accuracy_score(k_pred_y, k_y_test))
        feat_import.append(clf.feature_importances_)
    avg_accur = sum(accur_list)/len(accur_list)
    avg_feat_im = sum(feat_import)/len(accur_list)
    for i in range(len(accur_list)):
        sum_of_acc = sum_of_acc +(accur_list[0] - avg_accur)**2
    std = math.sqrt((sum_of_acc)/len(accur_list))
    print("The average accuracy is: ",avg_accur, "and the std is: ", std)
    return avg_feat_im

#test all features:
def test_all_features(clf, X_train, y_train):
    accur_list = []
    for train_index, test_index in kf.split(X_train):
        #print("TRAIN:", train_index, "TEST:", test_index)
        k_X_train, k_X_test = X_train.iloc[train_index], X_train.iloc[test_index]
        k_y_train, k_y_test = y_train.iloc[train_index], y_train.iloc[test_index]
        k_X_train, k_y_train = sm.fit_sample(k_X_train, k_y_train)
        clf.fit(k_X_train,k_y_train)
        k_pred_y = clf.predict(k_X_test)
        accur_list.append(accuracy_score(k_pred_y, k_y_test))
    avg_accur = sum(accur_list)/len(accur_list)
    print(avg_accur)
    

def get_scores(score_func, X, Y, X_train):
    k_best = SelectKBest(score_func = score_func, k = X.shape[1])
    fit = k_best.fit(X,Y)
    scores = pd.Series(index = X_train.columns, data = fit.scores_)
    return scores


def get_feat_acc (X_train, y_train, indices, classifier):
    list_of_acc = []
    list_of_std = []
    for i in range(len(indices)):
        print('This is index: ',i)
        acc, std = filter_features(X_train,y_train,index = indices[i], clf=classifier)
        list_of_acc.append(acc)
        list_of_std.append(std)
    list_of_acc = pd.DataFrame(list_of_acc)
    list_of_std = pd.DataFrame(list_of_std)

    return list_of_acc, list_of_std

def main():
    path = "data_cleaned.csv"
    data = pd.read_csv(path)
    #print(data.shape)
    x = data.iloc[:,3:124]
    y = data['readmitted']

    #Column droping 
    x = x.drop(['race_Other', 'gender_Male', 'gender_Unknown/Invalid',], axis = 1)
    #Need to fix admission type id
    #need to fix discharge disposition id
    #need to fix admission source id


    #Values: “up” if the dosage was increased during the encounter, 
    #“down” if the dosage was decreased, 
    #“steady” if the dosage did not change, and “no” if the drug was not prescribed
    #Removed'glimepiride-pioglitazone_Steady', and 'metformin-pioglitazone_Steady', 
    x = x.drop(['diag_1_Other', 'diag_2_Other', 'diag_3_Other', 'metformin_No', 
            'repaglinide_No', 'nateglinide_No','chlorpropamide_No',
            'glimepiride_No', 'acetohexamide_No', 'glipizide_No', 'glyburide_No',
            'tolbutamide_No', 'pioglitazone_No', 'rosiglitazone_No', 'acarbose_No','miglitol_No',
            'troglitazone_No','tolazamide_No','insulin_No', 'glyburide-metformin_No',
            'glipizide-metformin_No','glimepiride-pioglitazone_Steady',
            'metformin-pioglitazone_Steady',              
            'glimepiride-pioglitazone_No','metformin-rosiglitazone_No',
            'metformin-pioglitazone_No', 'examide_No', 'citoglipton_No'], axis = 1)

    x = x.drop(['admission_type_id'], axis = 1)
    #1:Referal
    #2:emergency room
    #4:transfer
    #3:other

    x['admission_source_id'][x['admission_source_id'] == 3] = 99
    x['admission_source_id'][x['admission_source_id'] == 2] = 99
    x['admission_source_id'][x['admission_source_id'] == 1] = 99
    x['admission_source_id'][x['admission_source_id'] == 7] = 98
    x['admission_source_id'][x['admission_source_id'] == 20] = 97
    x['admission_source_id'][x['admission_source_id'] == 17] = 97
    x['admission_source_id'][x['admission_source_id'] == 8] = 97
    x['admission_source_id'][x['admission_source_id'] == 9] = 97
    x['admission_source_id'][x['admission_source_id'] == 11] = 97
    x['admission_source_id'][x['admission_source_id'] == 13] = 97
    x['admission_source_id'][x['admission_source_id'] == 14] = 97
    x['admission_source_id'][x['admission_source_id'] == 4] = 96
    x['admission_source_id'][x['admission_source_id'] == 6] = 96
    x['admission_source_id'][x['admission_source_id'] == 5] = 96
    x['admission_source_id'][x['admission_source_id'] == 4] = 96
    x['admission_source_id'][x['admission_source_id'] == 10] = 96
    x['admission_source_id'][x['admission_source_id'] == 22] = 96
    x['admission_source_id'][x['admission_source_id'] == 25] = 96
    x['admission_source_id'][x['admission_source_id'] == 99] = 1
    x['admission_source_id'][x['admission_source_id'] == 98] = 2
    x['admission_source_id'][x['admission_source_id'] == 97] = 3
    x['admission_source_id'][x['admission_source_id'] == 96] = 4

    #1: home
    #2: other
    x['discharge_disposition_id'][x['discharge_disposition_id'] == 1] = 99
    x['discharge_disposition_id'][x['discharge_disposition_id'] == 6] = 99
    x['discharge_disposition_id'][x['discharge_disposition_id'] == 8] = 99
    x['discharge_disposition_id'][x['discharge_disposition_id'] != 99] = 98
    x['discharge_disposition_id'][x['discharge_disposition_id'] == 99] = 1
    x['discharge_disposition_id'][x['discharge_disposition_id'] == 98] = 2


    #Train test split:
    X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=.20, random_state=42)  

    #Feature Importance Section:
    #SMOTE Portion
    sm = SMOTE(random_state = 42, ratio = 1)
    print(X_train.shape)
    X_res, y_res = sm.fit_sample(X_train, y_train)
    print(X_res.shape)

    print(y_train.count())
    print(y_train[y_train == 1].count())
    print(y_train[y_train == 0].count()/y_train[y_train == 1].count())

    y_res = pd.DataFrame(y_res)
    print(y_res.count())
    print(y_res[y_res == 1].count())
    print(y_res[y_res == 0].count()/y_res[y_res == 1].count())

    #Declare Classifiers 
    clf1 = LogisticRegression(solver = 'liblinear')
    clf2 = Perceptron(tol=1e-3, random_state=42)
    clf3 = GaussianNB()
    clf4 = SVC(gamma='auto')
    clf5 = RandomForestClassifier(n_estimators='warn', criterion='gini', max_depth=None, 
                              min_samples_split=2, min_samples_leaf=1, 
                              min_weight_fraction_leaf=0.0, max_features='auto', 
                              max_leaf_nodes=None, min_impurity_decrease=0.0, 
                              min_impurity_split=None, bootstrap=True, oob_score=False, 
                              n_jobs=None, random_state=None, verbose=0, warm_start=False, 
                              class_weight=None)
    clf6 = KNeighborsClassifier(n_neighbors=5)

    min_max_scaler = preprocessing.MinMaxScaler()

    #Declare kf
    kf = KFold(n_splits=5, random_state=42, shuffle=False)

    #Get importance scores: 
    chi2_score = get_scores(chi2,X_res, y_res, X_train)
    f_classif_score = get_scores(f_classif, X_res, y_res, X_train)
    mutual_info_score = get_scores (mutual_info_classif, X_res, y_res, X_train)
    rand_for_score = get_avg_feat_importance(clf5, X_train, y_train)
   

    #Create Dataframe
    scores = pd.DataFrame(dict(Chi2 = chi2_score, F_class_if = f_classif_score, MI = mutual_info_score, Rand_For = rand_for_score)).reset_index()
    feature_names = scores['index']
    scores = pd.DataFrame(min_max_scaler.fit_transform(scores.iloc[:,1:]))
    scores.columns = ['Chi2','f_class_if','Mutual_Info','Rand_Forest']
    scores['Features'] = feature_names
    cols = scores.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    scores = scores[cols]
    scores['mean'] = scores.mean(axis = 1)
    scores['Chi2_Rank'] = scores['Chi2'].rank(ascending = False)
    scores['f_class_if Rank'] = scores['f_class_if'].rank(ascending = False)
    scores['Mutual_Info_Rank'] = scores['Mutual_Info'].rank(ascending = False)
    scores['Rand_For_Rank'] = scores['Rand_Forest'].rank(ascending = False)
    
    scores.to_csv("Feature_Select_Scores.csv")
    #print(scores)
    
    
    #filtering method
    
    index_1 = scores['Chi2'].sort_values(ascending = False).index
    index_2 = scores['f_class_if'].sort_values(ascending = False).index
    index_3 = scores['Mutual_Info'].sort_values(ascending = False).index
    index_4 = scores['Rand_Forest'].sort_values(ascending = False).index
    index_5 = scores['mean'].sort_values(ascending = False).index

    indices = []
    indices.append(index_1)
    indices.append(index_2)
    indices.append(index_3)
    indices.append(index_4)
    indices.append(index_5)
    #print(indices)
    
    print(scores.sort_values('mean', ascending=False))
    
    logit_acc, logit_std = get_feat_acc(X_train, y_train, indices, clf1)
    logit_acc.to_csv('logit_accuracy.csv')
    logit_std.to_csv('logit_standard_dev.csv')
    percep_acc, percep_std = get_feat_acc(X_train, y_train, indices, clf2)
    percep_acc.to_csv('percept_accuracy.csv')
    percep_std.to_csv('percept_standard_dev.csv')
    NB_acc, NB_std = get_feat_acc(X_train, y_train, indices, clf3)
    NB_acc.to_csv('NB_accuracy.csv')
    NB_std.to_csv('NB_standard_dev.csv')
    SVC_acc, SVC_std = get_feat_acc(X_train, y_train, indices, clf4)
    SVC_acc.to_csv('SVC_accuracy.csv')
    SVC_std.to_csv('SVC_standard_dev.csv')
    Rand_For_acc, Rand_For_std = get_feat_acc(X_train, y_train, indices, clf5)
    Rand_For_acc.to_csv('Rand_For_accuracy.csv')
    Rand_For_std.to_csv('Rand_For_standard_dev.csv')
    KNN_acc, KNN_std = get_feat_acc(X_train, y_train, indices, clf6)
    KNN_acc.to_csv('KNN_accuracy.csv')
    KNN_std.to_csv('KNN_standard_dev.csv')
    
    
if __name__ == "__main__":
    main()
    
