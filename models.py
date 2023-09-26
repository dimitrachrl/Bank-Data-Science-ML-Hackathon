import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import xgboost as xgb
import pickle

dataset=pd.read_csv('final_train_set.csv')

X=dataset.drop('plan',axis=1)
y=dataset['plan']

cv=5

X_train,X_test, y_train ,y_test = train_test_split(X,y,random_state=0,test_size=0.20)

Lreg=LogisticRegression()
Dtree=DecisionTreeClassifier()
RForest=RandomForestClassifier()
knn=KNeighborsClassifier()
svc=SVC()
xgb_classifier=xgb.XGBClassifier()
ada=AdaBoostClassifier()


score_Lreg=cross_val_score(Lreg, X,y, cv=cv, scoring='f1_macro')


score_Dtree=cross_val_score(Dtree, X, y, cv=cv, scoring='f1_macro')


score_RForest=cross_val_score(RForest, X, y, cv=cv, scoring='f1_macro')


score_knn=cross_val_score(knn, X, y, cv=cv, scoring='f1_macro')


score_svc=cross_val_score(svc, X, y, cv=cv, scoring='f1_macro')


score_xgb_classifier=cross_val_score(xgb_classifier, X, y, cv=5, scoring='f1_macro')


score_ada=cross_val_score(ada, X, y, cv=5, scoring='f1_macro')



print('Lreg score: ',score_Lreg.mean())
print('Dtree score: ',score_Dtree.mean())
print('RForest: ',score_RForest.mean())
print('Knn score: ',score_knn.mean())
print('SVC score: ',score_svc.mean())
print('XGBoost score :',score_xgb_classifier.mean())
print('AdaBoost score: ',score_ada.mean())



################################ HYPER PARAMETER TUNING ################################


################ LOGISTIC REGRESSION ################ 

Lreg_parameters ={'penalty' : ['l1','l2'], 'C':[10,1, 0.1, 0.01, 0.001, 0.0001], 'tol':[0.1, 0.001, 0.00001]}

grid_Lreg=GridSearchCV(Lreg, cv=5, scoring='accuracy', param_grid=Lreg_parameters, n_jobs=-1)
grid_Lreg.fit(X_train,y_train)

best_Lreg=grid_Lreg.best_estimator_
best_Lreg.fit(X,y)
y_best_Lreg=best_Lreg.predict(X_test)

print(classification_report(y_test,y_best_Lreg))

'''
grid_Dtree=GridSearchCV(Dtree, cv=5, scoring='accuracy', param_grid=Lreg_parameters, n_jobs=-1)
grid_Dtree.fit(X_train,y_train)

best_Dtree=grid_Dtree.best_estimator_
best_Dtree.fit(X_train,y_train)
y_best_Dtree=best_Dtree.predict(X_test)

print(classification_report(y_test,y_best_Dtree))
'''



################ RANDOM FOREST ################ 

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt', 'log2']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
#max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10,15,20]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2,5,15,20]
# Method of selecting samples for training each tree
bootstrap = [True, False]# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}



random_RForest=RandomizedSearchCV(RForest ,param_distributions = random_grid, cv=cv, scoring="f1_macro", n_jobs=-1, verbose=1)
random_RForest.fit(X, y)


best_RForest=random_RForest.best_estimator_
best_RForest.fit(X_train,y_train)
y_best_RForest=best_RForest.predict(X_test)


print(classification_report(y_test,y_best_RForest))


################ KNN ################ 

knn_parameters={'n_neighbors':range(0,40)}

grid_knn=GridSearchCV(knn, cv=5, scoring='f1_macro', param_grid=knn_parameters, n_jobs=-1)
grid_knn.fit(X, y)

best_knn=grid_knn.best_estimator_
best_knn.fit(X_train,y_train)
y_best_knn=best_knn.predict(X_test)


print(classification_report(y_test,y_best_knn))




################ XGBOOST ################ 

n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
# Various learning rate parameters
learning_rate = ['0.05','0.1', '0.2','0.3','0.5','0.6']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
# max_depth.append(None)
#Subssample parameter values
subsample=[0.7,0.6,0.8]
# Minimum child weight parameters
min_child_weight=[3,4,5,6,7]

random_grid = {'n_estimators': n_estimators,
               'learning_rate': learning_rate,
               'max_depth': max_depth,
               'subsample': subsample,
               'min_child_weight': min_child_weight}



xgb_randomized=RandomizedSearchCV(xgb_classifier ,param_distributions = random_grid, cv=cv, scoring="f1_macro", n_jobs=-1, verbose=1)
xgb_randomized.fit(X,y)

best_xgb_score=xgb_randomized.best_score_


best_xgb=xgb_randomized.best_estimator_
best_xgb.fit(X_train,y_train)
y_best_xgb=best_xgb.predict(X_test)

print(classification_report(y_test,y_best_xgb)) 






model='model.pkl'

file=open(model, 'wb')
pickle.dump(best_xgb, file)



