import pandas as pd
# Logistic regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
%matplotlib inline
# reading data
df= pd.read_csv('data.csv')
df.Class= df.Class.astype('category')
# Partion the dependent and independent variables
y = df.Class
X= df.drop(['Class'], axis = 1)
# partion the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
# standardize the variables
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
# create a random forest
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)
# classification report
print(classification_report(y_test, pred_rfc))
# Determining the accuracy of the model
RnAcc =accuracy_score(pred_rfc, y_test)*100
print("Random Forest accuracy score: ",RnAcc.round(2), "%")
# confusion matrix
print(confusion_matrix(y_test, pred_rfc))
# Cross validation using Random forest
rfc_eval = cross_val_score(estimator = rfc, X = X_train, y = y_train, cv = 10)
print(rfc_eval)
rfc_eval.max()*100
# AUC ROC
prob_pred_rfc=rfc.predict_proba(X_test)
from sklearn.metrics import roc_curve, auc
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, pred_rfc, pos_label=3)
roc_auc = auc(false_positive_rate, true_positive_rate)
roc_auc
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
# SUpport vector
svc = SVC()
svc.fit(X_train, y_train)
pred_svc = svc.predict(X_test)
# classificaition report
print(classification_report(y_test, pred_svc))
SVAcc =accuracy_score(pred_svc, y_test)*100
print("Support Vector accuracy score: ",SVAcc.round(2), "%")
# Fine tuning
param = {
    'C': [0.1,0.8,0.9,1,1.1,1.2,1.3,1.4],
    'kernel':['linear', 'rbf'],
    'gamma' :[0.1,0.8,0.9,1,1.1,1.2,1.3,1.4]
}
grid_svc = GridSearchCV(svc, param_grid=param, scoring='accuracy', cv=10)
grid_svc.fit(X_train, y_train)
# determining the best parameters
grid_svc.best_params_
svc2 = SVC(C = 0.8, gamma =  0.1, kernel= 'linear', probability=True)
svc2.fit(X_train, y_train)
pred_svc2 = svc2.predict(X_test)
print(classification_report(y_test, pred_svc2))
SVAcc =accuracy_score(pred_svc2, y_test)*100
print("Support Vector accuracy score: ",SVAcc.round(2), "%")
prob_pred_rfc=svc2.predict_proba(X_test)
# AUC score
from sklearn.metrics import roc_auc_score
#prob_pred_svc2=
auc_roc=roc_auc_score(y_test,prob_pred_rfc, multi_class='ovr')
auc_roc
from sklearn.metrics import roc_curve, auc
false_positive_rate1, true_positive_rate1, thresholds = roc_curve(y_test, pred_svc2, pos_label=3)
roc_auc1 = auc(false_positive_rate, true_positive_rate)
roc_auc1
# ROC PLOT
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate1,true_positive_rate1, color='red',label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
# creating the both the plots for comparison
fig, axs = plt.subplots(2)
fig.suptitle('Vertically stacked subplots')
axs[1].plot(false_positive_rate1,true_positive_rate1, color='red',label = 'AUC = %0.2f' % roc_auc1)
axs[1].legend(loc = 'lower right')
axs[1].plot([0, 1], [0, 1],linestyle='--')
axs[1].axis('tight')

axs[0].plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
axs[0].legend(loc = 'lower right')
axs[0].plot([0, 1], [0, 1],linestyle='--')
axs[0].axis('tight')

