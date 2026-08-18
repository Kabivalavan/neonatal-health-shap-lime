import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

df=pd.read_csv("data/neonatal_health_synthetic.csv")
X=df.drop(columns=["high_risk"]); y=df["high_risk"]
numeric=X.select_dtypes(include="number").columns.tolist()
categorical=X.select_dtypes(exclude="number").columns.tolist()
pre=ColumnTransformer([("num",StandardScaler(),numeric),("cat",OneHotEncoder(handle_unknown="ignore",sparse_output=False),categorical)])
model=RandomForestClassifier(n_estimators=300,max_depth=10,min_samples_leaf=3,class_weight="balanced",random_state=42,n_jobs=-1)
pipeline=Pipeline([("preprocessor",pre),("model",model)])
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.20,stratify=y,random_state=42)
pipeline.fit(X_train,y_train)
joblib.dump(pipeline,"src/neonatal_risk_model.joblib")
