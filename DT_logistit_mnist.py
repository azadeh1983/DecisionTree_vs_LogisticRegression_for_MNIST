
from scipy.stats import randint
from sklearn.model_selection import  RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
import pandas as pd




#load dataset------------------------------------------
#download mnist.npz from google drive
path = 'mnist.npz'
with np.load(path, allow_pickle=True) as f:
    x_train1, y_train = f['x_train'], f['y_train']
    x_test1, y_test = f['x_test'], f['y_test']

# reshape samples of train and test
x_train = x_train1.reshape(x_train1.shape[0], -1)

x_test = x_test1.reshape(x_test1.shape[0], -1)


#create my model of LogisticRegression---------------
lr=LogisticRegression()
lr.fit(x_train,y_train)

#predict for x_test
y_pred=lr.predict(x_test)

#create confusion matrix
cm=confusion_matrix(y_test,y_pred)


#normalize confusion mtrix
cm=normalize(cm,norm='l1',axis=1)

#make a Data Frame from confusion matrix with labels
cm_df=pd.DataFrame(cm,columns=[0,1,2,3,4,5,6,7,8,9],index=[0,1,2,3,4,5,6,7,8,9])
print('çonfusion matrix for LogisticRegression')
print(cm_df)

score=lr.score(x_test,y_test)

#create my model DecisionTree-----------------------
params={'max_depth':[None,3],'max_features':randint(1,9),'min_samples_leaf':randint(1,9)}
tree=DecisionTreeClassifier()
tree_cv=RandomizedSearchCV(tree,params,cv=5)
tree_cv.fit(x_train,y_train)

y_pred_DT=tree_cv.predict(x_test)
cm_DT=confusion_matrix(y_test,y_pred_DT)
#normalize confusion mtrix
cm_DT=normalize(cm_DT,norm='l1',axis=1)

#make a Data Frame from confusion matrix with labels
cm_DT_df=pd.DataFrame(cm_DT,columns=[0,1,2,3,4,5,6,7,8,9],index=[0,1,2,3,4,5,6,7,8,9])
print('çonfusion matrix for Decision Tree')
print(cm_DT_df)

# print(tree_cv.best_params_)
score_tree=tree_cv.score(x_test,y_test)
print(f"score of Decision LogisticRegression : {score}, score of Tree: {score_tree}")
