import pandas as  pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
import numpy as np
#veriyi hazırlama
pf=pd.read_csv("../../Datasets/Cancer.csv")
X=pf.drop(['Unnamed: 32',"id","diagnosis"],axis=1)
Y=np.array(pd.get_dummies(pf['diagnosis'], drop_first=True)).reshape(X.shape[0])

#veriyi bölme
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.21,random_state=42)

#modeli kurma

Catboost=CatBoostClassifier(verbose=False)

Catboost.fit(X_train,y_train)

#modelden tahmin tapma
pred=Catboost.predict(X_test)

#ilkel başarı değeri
print(f"İlkel başarı değeri : {accuracy_score(y_test,pred)}")


#hiperparametre seçelim

hiperparams={'depth': np.arange(2,10,2),
             'learning_rate': [0.0001,0.001,0.01,0.1,1],
             'iterations': np.arange(200,1000,200)}

model_cv=GridSearchCV(Catboost,hiperparams,cv=10,n_jobs=-1).fit(X_train,y_train)
print(model_cv.best_params_)

# Model tunnig
model_tunned=CatBoostClassifier(learning_rate=model_cv.best_params_['learning_rate'],
                           iterations=model_cv.best_params_['iterations'],
                           verbose=False,
                           depth=model_cv.best_params_['depth']).fit(X_train,y_train)
                        

pred_tunned=model_tunned.predict(X_test)
print(f"İlkel başarı değeri : {accuracy_score(y_test,pred)}")
print(f"Tunned edilmiş başarı değeri : {accuracy_score(y_test,pred_tunned)}")