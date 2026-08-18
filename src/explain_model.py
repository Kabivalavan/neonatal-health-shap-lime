import numpy as np, pandas as pd, shap
from sklearn.linear_model import Ridge
import joblib
pipe=joblib.load("src/neonatal_risk_model.joblib")
df=pd.read_csv("data/neonatal_health_synthetic.csv"); X=df.drop(columns=["high_risk"])
Xt=pipe.named_steps["preprocessor"].transform(X); names=pipe.named_steps["preprocessor"].get_feature_names_out()
explainer=shap.TreeExplainer(pipe.named_steps["model"]); values=explainer.shap_values(Xt)
print("SHAP explanation generated for",len(Xt),"records")
