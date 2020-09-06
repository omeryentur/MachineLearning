import pandas as  pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore",category=ConvergenceWarning)
scaler = StandardScaler()
#veriyi hazırlama
pf=pd.read_csv("../../Datasets/Cancer.csv")
X=pf.drop(['Unnamed: 32',"id","diagnosis"],axis=1)
Y=np.array(pd.get_dummies(pf['diagnosis'], drop_first=True)).reshape(X.shape[0])

#veriyi bölme
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.21,random_state=42)
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)
#modeli kurma

NeuralNetwork=MLPClassifier(solver='lbfgs')

NeuralNetwork.fit(X_train,y_train)

#modelden tahmin tapma
pred=NeuralNetwork.predict(X_test)

#ilkel başarı değeri
print(f"İlkel başarı değeri : {accuracy_score(y_test,pred)}")


#hiperparametre seçelim

hiperparams={'hidden_layer_sizes': [(3,5),(8,16),(32,64),(64,128),(128,128),(16,32,16)],
             'max_iter': np.arange(1000,2000,250),
             'solver': ['adam','lbfgs','sgd']}

model_cv=GridSearchCV(NeuralNetwork,hiperparams,cv=10,n_jobs=-1).fit(X_train,y_train)
print(model_cv.best_params_)

# Model tunnig
model_tunned=MLPClassifier(hidden_layer_sizes=model_cv.best_params_['hidden_layer_sizes'],
                           solver=model_cv.best_params_['solver'],
                           max_iter=model_cv.best_params_['max_iter']).fit(X_train,y_train)
                        

pred_tunned=model_tunned.predict(X_test)
print(f"İlkel başarı değeri : {accuracy_score(y_test,pred)}")
print(f"Tunned edilmiş başarı değeri : {accuracy_score(y_test,pred_tunned)}")