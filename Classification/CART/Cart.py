import pandas as  pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np
#veriyi hazırlama
pf=pd.read_csv("../../Datasets/Cancer.csv")
X=pf.drop(['Unnamed: 32',"id","diagnosis"],axis=1)
Y=np.array(pd.get_dummies(pf['diagnosis'], drop_first=True)).reshape(X.shape[0])

#veriyi bölme
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.21,random_state=42)

#modeli kurma

Cart=DecisionTreeClassifier()

Cart.fit(X_train,y_train)

#modelden tahmin tapma
pred=Cart.predict(X_test)

#ilkel başarı değeri
print(f"İlkel başarı değeri : {accuracy_score(y_test,pred)}")


#hiperparametre seçelim

hiperparams={'min_samples_leaf': np.arange(1,20,2),
             'max_depth': np.arange(2,20,2),
             'min_samples_split': np.arange(2,20,2)}

model_cv=GridSearchCV(Cart,hiperparams,cv=10,n_jobs=-1).fit(X_train,y_train)
print(model_cv.best_params_)

# Model tunnig
model_tunned=DecisionTreeClassifier(min_samples_leaf=model_cv.best_params_['min_samples_leaf'],
                           min_samples_split=model_cv.best_params_['min_samples_split'],
                           max_depth=model_cv.best_params_['max_depth']).fit(X_train,y_train)
                        

pred_tunned=model_tunned.predict(X_test)
print(f"İlkel başarı değeri : {accuracy_score(y_test,pred)}")
print(f"Tunned edilmiş başarı değeri : {accuracy_score(y_test,pred_tunned)}")