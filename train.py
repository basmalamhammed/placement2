from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from dataprocessing import build_preprocessor,load_data,split_XY


BASE_DIR=Path(__file__).parent
MODELS_DIR=BASE_DIR/"model"
MODELS_DIR.mkdir(exist_ok=True)

df= load_data()
X,Y=split_XY(df)

X_train, X_test, Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=42)


models={
    "Logistic Regression":LogisticRegression(max_iter=1000),
    "KNN":KNeighborsClassifier(n_neighbors=7),
    "Naive Bayes":GaussianNB(),
    "Decision Tree ":DecisionTreeClassifier(max_depth=4,random_state=42),
    "Random Forest":RandomForestClassifier(n_estimators=200,random_state=42),
    "SVM":SVC(probability=True),  
    }

result=[]
trained={}
for name,clf in models.items():
    pipe=Pipeline([
       ( "processor",build_preprocessor()),
       ("model",clf),
    ])

    pipe.fit(X_train,Y_train)
    predictions=pipe.predict(X_test)

    result.append({
        "model":name,
        "Acuuracy":accuracy_score(Y_test,predictions),
        "precision":precision_score(Y_test,predictions),
        "F1":f1_score(Y_test,predictions),
        "Recall":recall_score(Y_test,predictions),
    })
    trained[name]= pipe
    print(f"done: {name}")

results_df = pd.DataFrame(result).sort_values("F1", ascending=False).reset_index(drop=True)
print("\n", results_df.round(3).to_string(index=False))

best_name = results_df.loc[0, "model"]
print(f"\nBest model (by F1): {best_name}")

# ---------- 5) save ----------
results_df.to_csv(MODELS_DIR / "results.csv", index=False)
joblib.dump(trained[best_name], MODELS_DIR / "placement_model.joblib")
(MODELS_DIR / "best_model_name.txt").write_text(best_name)



