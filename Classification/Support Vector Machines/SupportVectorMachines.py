import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings("ignore",category=FutureWarning)
warnings.filterwarnings("ignore",category=DeprecationWarning)


#veriyi hazırlama
pf=pd.read_csv("../../Datasets/Cancer.csv")
X=pf.drop(['Unnamed: 32',"id","diagnosis"],axis=1)
Y=np.array(pd.get_dummies(pf['diagnosis'], drop_first=True)).reshape(X.shape[0])

#veriyi bölme
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.21,random_state=42)

#modeli kurma

svc=SVC(gamma='auto')

svc.fit(X_train,y_train)

#modelden tahmin tapma
pred=svc.predict(X_test)

#ilkel başarı değeri
print(f"İlkel başarı değeri : % {accuracy_score(y_test,pred)*100}")


#hiperparametre seçelim
# C parametresi SVM de ceza parametresidir.
hiperparams={'C': np.arange(1,100,1),"gamma": ['auto','scale']}

model_cv=GridSearchCV(svc,hiperparams,cv=10,n_jobs=-1).fit(X_train,y_train)

print(f"en iyi parametreler {model_cv.best_params_}")

# Model tunnig
model_tunned=SVC(C=model_cv.best_params_['C'],gamma=model_cv.best_params_['gamma']).fit(X_train,y_train)

pred_tunned=model_tunned.predict(X_test)

print(f"Tunned edilmiş başarı değeri : % {accuracy_score(y_test,pred_tunned)*100}")