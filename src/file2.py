import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub
dagshub.init(repo_owner='kumarashutoshbtech2023', repo_name='mlflow_', mlflow=True)
mlflow.set_tracking_uri('https://dagshub.com/kumarashutoshbtech2023/mlflow_.mlflow')
from mlflow.models.signature import infer_signature
wine=load_wine()
x=wine.data
y=wine.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=42)
max_depth=5
n_estimators=10
mlflow.set_experiment('mlflow')
ip_example=pd.DataFrame(x_train[:5],columns=wine.feature_names)
signature=infer_signature(x_train,y_train)
with mlflow.start_run():
    rf=RandomForestClassifier(max_depth=max_depth,n_estimators=n_estimators,random_state=42)
    rf.fit(x_train,y_train)
    y_pred=rf.predict(x_test)
    accuracy=accuracy_score(y_test,y_pred)

    mlflow.log_metric('accuracy',accuracy)
    mlflow.log_param('max_depth',max_depth)
    mlflow.log_param('n_estimators',n_estimators)

    print(accuracy)
    cm=confusion_matrix(y_test,y_pred)
    
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.savefig("Confusion-matrix.png")
    mlflow.log_artifact("Confusion-matrix.png") 
    mlflow.log_artifact(__file__)

    # mlflow.sklearn.log_model(sk_model=rf, name="Random-Forest-Model", input_example=ip_example,signature=signature)
