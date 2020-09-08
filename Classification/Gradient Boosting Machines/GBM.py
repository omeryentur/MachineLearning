import pandas as  pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import numpy as np
#veriyi hazırlama
pf=pd.read_csv("../../Datasets/Cancer.csv")
X=pf.drop(['Unnamed: 32',"id","diagnosis"],axis=1)
Y=np.array(pd.get_dummies(pf['diagnosis'], drop_first=True)).reshape(X.shape[0])

#veriyi bölme
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.21,random_state=42)

#modeli kurma

Gbm=GradientBoostingClassifier()

Gbm.fit(X_train,y_train)

#modelden tahmin tapma
pred=Gbm.predict(X_test)

#ilkel başarı değeri
print(f"İlkel başarı değeri : {accuracy_score(y_test,pred)}")


#hiperparametre seçelim

hiperparams={'max_depth': np.arange(2,10,2),
             'learning_rate': [0.0001,0.001,0.01,0.1,1],
             'n_estimators': np.arange(200,1000,200)}

model_cv=GridSearchCV(Gbm,hiperparams,cv=10,n_jobs=-1,verbose=2).fit(X_train,y_train)
print(model_cv.best_params_)

# Model tunnig
model_tunned=GradientBoostingClassifier(learning_rate=model_cv.best_params_['learning_rate'],
                           n_estimators=model_cv.best_params_['n_estimators'],
                           max_depth=model_cv.best_params_['max_depth']).fit(X_train,y_train)
                        

pred_tunned=model_tunned.predict(X_test)
print(f"İlkel başarı değeri : {accuracy_score(y_test,pred)}")
print(f"Tunned edilmiş başarı değeri : {accuracy_score(y_test,pred_tunned)}")