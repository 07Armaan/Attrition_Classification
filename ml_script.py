# %%
import numpy as np
import pandas as pd

# %% [markdown]
# # Data Injection

# %%
df = pd.read_csv("attrition_dataset.csv")

# %% [markdown]
# # Data Inspection and basic data improving

# %%
df.head()

# %% [markdown]
# * Employee ID column should be removed

# %%
df.shape

# %% [markdown]
# * 59.5+ rows
# * 24 columns/features

# %% [markdown]
# df.columns

# %%
df.duplicated().sum()

# %% [markdown]
# * No duplicate rows

# %%
df.isnull().sum()

# %% [markdown]
# * No null / empty values

# %%
df = df.drop(columns=["Employee ID"])

# %%
df.describe()

# %% [markdown]
# * No logical mistakes in numerical data

# %%
df.dtypes

# %% [markdown]
# * All column has correct datatype

# %% [markdown]
# *Numeric Columns*

# %%
df.select_dtypes(include=["int"]).columns

# %% [markdown]
# *Categorical / Object Columns*

# %%
df.select_dtypes(include=["object"]).drop(columns=["Attrition"]).columns

# %%
num_cols = df.select_dtypes(include=["int"]).columns
cat_cols = df.select_dtypes(include=["object"]).drop(columns=["Attrition"]).columns
tar_col = df["Attrition"]

# %% [markdown]
# # Exploratory Data Analysis

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# %% [markdown]
# **Univariate Analysis**

# %% [markdown]
# *Numeric Columns*

# %%
right_skewed_cols = []
left_skewed_cols = []
nrml_skewed_cols = []
for col in num_cols:
    sns.kdeplot(x=col,data=df)
    plt.title(col)
    plt.show()
    print(df[col].skew())
    if df[col].skew()>=0.6:
        right_skewed_cols.append(col)
    elif df[col].skew()<=-0.6:
        left_skewed_cols.append(col)
    else:
        nrml_skewed_cols.append(col)

# %%
print(f"Right skewed cols: {right_skewed_cols}")
print(f"Left skewed cols: {left_skewed_cols}")
print(f"Nrml skewed cols: {nrml_skewed_cols}")

# %% [markdown]
# * The numeric cols are not >= skewed to -0.6
# * 3 cols are skewed right
# * 4 cols are almost nrml skewed

# %%
for col in num_cols:
    sns.boxplot(x=col,data=df)
    plt.title(col)
    plt.show()

# %% [markdown]
# * Less outliers present in "Years at Company" col
# * Many outliers present in "Monthly Income" col

# %% [markdown]
# *Categorical Columns*

# %%
for col in cat_cols:
    sns.countplot(y=col,data=df)
    plt.title(col)
    plt.show()

# %% [markdown]
# *Target Column*

# %%
sns.countplot(x=tar_col,data=df)
plt.title("Target Column")
plt.show()
print(tar_col.value_counts())
print("\n\n")
print(tar_col.value_counts(normalize=True)*100)

# %% [markdown]
# * ~5% data is imbalance only
# * So,no need of handling imbalance

# %% [markdown]
# **Bivariate Analysis**

# %% [markdown]
# *Num cols VS Target col*

# %%
for col in num_cols:
    sns.violinplot(x=col,y=tar_col,data=df)
    plt.title(col)
    plt.show()

# %% [markdown]
# *Cat cols VS Target col*

# %%
for col in cat_cols:
    sns.countplot(x=col,hue=tar_col,data=df)
    plt.title(col)
    plt.show()

# %% [markdown]
# **Multivariate Analysis**

# %%
sns.heatmap(df[num_cols].corr(),annot=True,fmt=".2f",cmap="Blues")

# %% [markdown]
# * No multicollinearity

# %% [markdown]
# # Handling Outliers

# %%
for col in num_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)

    iqr = q3-q1

    lower = q1-1.5*iqr
    upper = q3+1.5*iqr

    df[col] = df[col].clip(lower,upper)

# %% [markdown]
# # Feature Engg

# %%
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder,FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,precision_recall_curve,f1_score,precision_score,recall_score
import optuna

# %%
x = df.drop(columns=["Attrition"])
y = df["Attrition"]

# %% [markdown]
# # Splitting data

# %%
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=True)

# %% [markdown]
# # Encoding Target values

# %%
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

# %%
le.classes_

# %% [markdown]
# # Pipelines for each type of data

# %%
right_skew_pipeline = Pipeline(steps=[
    ("right_skew_imputer",SimpleImputer(strategy="mean")),
    ("right_skew_transformer",FunctionTransformer(np.log1p,feature_names_out='one-to-one')),
    ("right_skew_scaling",StandardScaler())
])

# %%
nrml_skew_pipeline = Pipeline(steps=[
    ("nrml_skew_imputer",SimpleImputer(strategy="mean")),
    ("nrml_skew_scaling",StandardScaler())
])

# %%
cat_pipeline = Pipeline(steps=[
    ("OHEncoder",OneHotEncoder(handle_unknown="ignore"))
])

# %% [markdown]
# # Preprocessing ColumnTransformer

# %%
preprocessing = ColumnTransformer(transformers=[
    ("right_skew_pipeline",right_skew_pipeline,right_skewed_cols),
    ("nrml_skew_pipeline",nrml_skew_pipeline,nrml_skewed_cols),
    ("cat_pipeline",cat_pipeline,cat_cols)
],remainder = "passthrough")

# %% [markdown]
# # Finding best model and best params (Optuna)

# %%
def objective(trial):
    model_name = trial.suggest_categorical("model", ["lr", "dt", "rf", "xgb"])

    if model_name == "lr":
        model = LogisticRegression(
        C=trial.suggest_float("C", 1e-3, 10, log=True),
        solver=trial.suggest_categorical("solver", ["lbfgs", "liblinear"]),
        max_iter=1000
    )

    elif model_name == "dt":
        model = DecisionTreeClassifier(
        max_depth=trial.suggest_int("max_depth", 3, 20),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
        criterion=trial.suggest_categorical("criterion", ["gini", "entropy"])
    )

    elif model_name == "rf":
        model = RandomForestClassifier(
        n_estimators=trial.suggest_int("n_estimators", 100, 500),
        max_depth=trial.suggest_int("max_depth", 5, 30),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
        max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        n_jobs=-1
    )

    elif model_name == "xgb":
        model = XGBClassifier(
        n_estimators=trial.suggest_int("n_estimators", 100, 500),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        max_depth=trial.suggest_int("max_depth", 3, 12),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        gamma=trial.suggest_float("gamma", 0, 5),
        reg_alpha=trial.suggest_float("reg_alpha", 0, 5),
        reg_lambda=trial.suggest_float("reg_lambda", 0, 5),
        use_label_encoder=False,
        eval_metric="logloss",
        n_jobs=-1,
        verbosity=0
    )
    pipeline = Pipeline(steps=[
        ("preprocessing",preprocessing),
        ("model",model)
    ])
    pipeline.fit(x_train,y_train)
    pipeline_pred = pipeline.predict(x_test)
    score = f1_score(y_test,pipeline_pred)
    return score

# %%
study = optuna.create_study(direction="maximize")
study.optimize(objective,n_trials=100)

# %%
params = study.best_params
params

# %%
model_name = params["model"]

if model_name == "lr":
    final_model = LogisticRegression(
        C=params["C"],
        solver=params["solver"],
        max_iter=1000
    )

elif model_name == "dt":
    final_model = DecisionTreeClassifier(
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        min_samples_leaf=params["min_samples_leaf"],
        criterion=params["criterion"]
    )

elif model_name == "rf":
    final_model = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        min_samples_leaf=params["min_samples_leaf"],
        max_features=params["max_features"],
        n_jobs=-1
    )

elif model_name == "xgb":
    final_model = XGBClassifier(
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        gamma=params["gamma"],
        reg_alpha=params["reg_alpha"],
        reg_lambda=params["reg_lambda"],
        use_label_encoder=False,
        eval_metric="logloss",
        n_jobs=-1,
        verbosity=0
    )


# %% [markdown]
# # Final Pipeline (preprocessing + best model with best params)

# %%
final_pipeline = Pipeline(steps=[
    ("preprocessing",preprocessing),
    ("final_model",final_model)
])

# %% [markdown]
# # Training full pipeline on training data

# %%
final_pipeline.fit(x_train,y_train)

# %% [markdown]
# # Prediction on test data

# %%
y_train_pred = final_pipeline.predict(x_train)
y_test_pred = final_pipeline.predict(x_test)

# %% [markdown]
# # Model Evaluation

# %% [markdown]
# **Accuracy**

# %%
train_acc = accuracy_score(y_train,y_train_pred)
print(f"Train acc: {train_acc}")
test_acc = accuracy_score(y_test,y_test_pred)
print(f"Test acc: {test_acc}")

# %%
train_cr = classification_report(y_train,y_train_pred,target_names=le.classes_)
print(f"Train cr: \n{train_cr}")
test_cr = classification_report(y_test,y_test_pred,target_names=le.classes_)
print(f"\n\nTest cr: \n{test_cr}")

# %%
train_cm = confusion_matrix(y_train,y_train_pred)
print(f"Train cm")
sns.heatmap(train_cm,annot=True,fmt=".2f",cmap="Blues")
plt.show()
test_cm = confusion_matrix(y_test,y_test_pred,)
print(f"\n\nTest cm")
sns.heatmap(test_cm,annot=True,fmt=".2f",cmap="Blues")
plt.show()

# %% [markdown]
# # Saving final_pipeline as .pkl file

# %%
import joblib
joblib.dump(final_pipeline,"final_pipeline.pkl")

# %% [markdown]
# # Model Explainability (Shap)

# %%
import shap

# %%
shap_preprocessor = final_pipeline.named_steps["preprocessing"]
shap_model = final_pipeline.named_steps["final_model"]

# %%
feature_names = []
for col in shap_preprocessor.get_feature_names_out():
    feature_names.append(col.split("__")[-1])

# %%
x_test_t = pd.DataFrame(
    shap_preprocessor.transform(x_test),
    columns=feature_names
)

# %%
explainer = shap.Explainer(shap_model)

# %%
shap_values = explainer(x_test_t)

# %%
shap.plots.bar(shap_values,max_display=len(feature_names))

# %%
shap.plots.beeswarm(shap_values,max_display=len(feature_names))

# %% [markdown]
# Mlflow

# %%
import mlflow

# %%
mlflow.set_experiment("Attrition_classification")
with mlflow.start_run(run_name="XGBC_21-06-2026"):
    mlflow.log_params(params)

    mlflow.log_metric("Train accuracy",train_acc)
    mlflow.log_metric("Test accuracy",test_acc)

    mlflow.log_metric("Train f1_score",f1_score(y_train,y_train_pred))
    mlflow.log_metric("Test f1_score",f1_score(y_test,y_test_pred))

    mlflow.log_metric("Train precision",precision_score(y_train,y_train_pred))
    mlflow.log_metric("Test precision",precision_score(y_test,y_test_pred))

    mlflow.log_metric("Train recall",recall_score(y_train,y_train_pred))
    mlflow.log_metric("Test recall",recall_score(y_test,y_test_pred))

    mlflow.sklearn.log_model(final_pipeline,"final_pipeline",serialization_format="cloudpickle")


# %%
best_run_id = "c3e2a7fabcd04f389d03cb4e5d8218ce"
best_run = f"runs:/{best_run_id}/final_pipeline"

# %%
result = mlflow.register_model(
    model_uri=best_run,
    name="attrition_classifier_registry"
)

# %%
production_model = mlflow.sklearn.load_model("models:/attrition_classifier_registry@production")
mlflow.sklearn.save_model(production_model,"production_model",serialization_format="cloudpickle")

# %%



