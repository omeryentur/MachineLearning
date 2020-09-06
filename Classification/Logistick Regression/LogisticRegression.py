import pandas as  pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
#veriyi hazırlama
pf=pd.read_csv("../../Datasets/Cancer.csv")
X=pf.drop(['Unnamed: 32',"id","diagnosis"],axis=1)
Y=np.array(pd.get_dummies(pf['diagnosis'], drop_first=True)).reshape(X.shape[0])
print((Y.shape))
#veriyi bölme
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.21,random_state=42)

#modeli kurma
logistic_model=LogisticRegression()

logistic_model.fit(X_train,y_train)

#modelden tahmin tapma
pred=logistic_model.predict(X_test)

#ilkel başarı değeri
print(f"İlkel başarı değeri : {accuracy_score(y_test,pred)}")