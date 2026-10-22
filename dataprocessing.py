from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).parent
DATA_PATH=BASE_DIR/"data"/"placement.csv"

TARGET="status"

NUM_COLS = ["ssc_p", "hsc_p", "degree_p", "etest_p"]

CAT_COLS = ["gender", "ssc_b", "hsc_b", "hsc_s", "degree_t", "workex"]


def load_data(path=DATA_PATH):
    df=pd.read_csv(path)
    df = df.drop(columns=["sl_no", "salary", "mba_p", "specialisation"], errors="ignore")
    df[TARGET]=df[TARGET].map({"Placed":1,"Not Placed":0})
    return df 

def split_XY(df):
    X=df.drop(TARGET,axis=1)
    Y=df[TARGET]
    return X,Y


def build_preprocessor():
    num_pipe=Pipeline(
        [
            ("imputer",SimpleImputer(strategy="median")),
            ("sclaer",StandardScaler()),

        ]
    )
    cat_pipe=Pipeline(
        [
            ("imputer",SimpleImputer(strategy="most_frequent")),
            ("onehot",OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer([
        ("num",num_pipe,NUM_COLS),
        ("cat", cat_pipe, CAT_COLS)
    ])






