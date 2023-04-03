# Databricks notebook source
# MAGIC %md 
# MAGIC You may find this series of notebooks at https://github.com/databricks-industry-solutions/fraud-orchestration. For more information about this solution accelerator, visit https://www.databricks.com/solutions/accelerators/fraud-detection.

# COMMAND ----------

# MAGIC %md
# MAGIC <img src=https://brysmiwasb.blob.core.windows.net/demos/dff/databricks_fsi_white.png width="600px">

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Databricks fraud framework - Model building
# MAGIC 
# MAGIC The financial service industry (FSI) is rushing towards transformational change to support new channels and services, delivering transactional features and facilitating payments through new digital channels to remain competitive. Unfortunately, the speed and convenience that these capabilities afford is a benefit to consumers and fraudsters alike. Building a fraud framework often goes beyond just creating a highly accurate machine learning model due ever changing landscape and customer expectation. Oftentimes it involves a complex decision science setup which combines rules engine with a need for a robust and scalable machine learning platform. In this series of notebook, we'll be demonstrating how `Delta Lake`, `MLFlow` and a unified analytics platform can help organisations combat fraud more efficiently
# MAGIC 
# MAGIC 
# MAGIC ---
# MAGIC + <a href="$./01_dff_model">STAGE1</a>: Integrating rule based with ML
# MAGIC + <a href="$./02_dff_orchestration">STAGE2</a>: Building a fraud detection model
# MAGIC ---
# MAGIC 
# MAGIC + <sri.ghattamaneni@databricks.com>
# MAGIC + <nikhil.gupta@databricks.com>
# MAGIC + <ricardo.portilla@databricks.com>

# COMMAND ----------

# DBTITLE 1,Interpretable ML Visualization Library Import
# MAGIC %pip install shap

# COMMAND ----------

# DBTITLE 1,Get relevant libraries
import numpy as np                   # array, vector, matrix calculations
import pandas as pd                  # DataFrame handling
import xgboost as xgb                # gradient boosting machines (GBMs)
import mlflow
import os
import mlflow.pyfunc
import mlflow.spark
import sklearn

# COMMAND ----------

# DBTITLE 1,Initiate data folder
# MAGIC %sh 
# MAGIC rm -r /dbfs/tmp/dff/
# MAGIC mkdir -p /dbfs/tmp/dff/
# MAGIC cp Fraud_final-1.csv /dbfs/tmp/dff/Fraud_final-1.csv

# COMMAND ----------

# DBTITLE 1,Persist Txn Flat Files to Delta Lake for Audit and Performance
# File location and type
raw_data_path = "/tmp/dff/delta_txns"

spark.read.option("inferSchema", "true") \
          .option("header", "true") \
          .option("delim", ",") \
          .csv("/tmp/dff/Fraud_final-1.csv") \
          .write \
          .format("delta") \
          .mode("overwrite") \
          .option("overwriteSchema", "true") \
          .save(raw_data_path)

# COMMAND ----------

# MAGIC %md 
# MAGIC Let's first define a outline for feature preprocessing and modeling. We will call the respective preprocessing and modeling functions after we have imported out data.

# COMMAND ----------

# This scaling code using the simple sklearn out-of-the-box scaler. It's used here for simplicity and re-used inside our PyFunc class
def preprocess_data(source_df,
                    numeric_columns,
                    fitted_scaler):
  '''
  Subset df with selected columns
  Use the fitted scaler to center and scale the numeric columns  
  '''
  res_df = source_df[numeric_columns].copy()
  
  ## scale the numeric columns with the pre-built scaler
  res_df[numeric_columns] = fitted_scaler.transform(res_df[numeric_columns])
  
  return res_df

# COMMAND ----------

# DBTITLE 1,PyFunc Wrapper for Fraud Model
class XGBWrapper(mlflow.pyfunc.PythonModel):
  '''
    XGBClassifier model with embedded pre-processing.
    
    This class is an MLflow custom python function wrapper around a XGB model.
    The wrapper provides data preprocessing so that the model can be applied to input dataframe directly.
    :Input: to the model is pandas dataframe
    :Output: predicted price for each listing

    The model declares current local versions of XGBoost and pillow as dependencies in its
    conda environment file.  
  '''
  def __init__(self,
               model,
               X,
               y,
               numeric_columns):
    
    self.model = model

    from sklearn.model_selection import train_test_split
    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.30, random_state=2019)
    self.numeric_columns = numeric_columns
    
    from sklearn.preprocessing import StandardScaler 
    #create a scaler for our numeric variables
    # only run this on the training dataset and use to scale test set later.
    scaler = StandardScaler()
    self.fitted_scaler = scaler.fit(self.X_train[self.numeric_columns])
    self.X_train_processed = preprocess_data(self.X_train, self.numeric_columns, self.fitted_scaler)
    self.X_test_processed  = preprocess_data(self.X_test, self.numeric_columns, self.fitted_scaler)

    def _accuracy_metrics(model, X, y):
      import sklearn
      from sklearn import metrics
      y_pred = model.predict_proba(X)[:,1]
      fpr, tpr, thresholds = sklearn.metrics.roc_curve(y, y_pred)
      self.auc = sklearn.metrics.auc(fpr, tpr)
      print("Model AUC is:", self.auc)

      return self.auc
    
    self.auc = _accuracy_metrics(model=self.model, X=self.X_test_processed, y=self.y_test )
    
    
  def predict(self, context, input):
    '''
      Generate predictions from the input df 
      Subset input df with selected columns
      Assess the model accuracy
      Use the fitted scaler to center and scale the numeric columns  
      :param input: pandas.DataFrame with numeric_columns to be scored. The
                   columns must has same schema as numeric_columns of X_train
     :return: numpy 1-d array as fraud probabilities 

    '''
    input_processed = self._preprocess_data(X=input, numeric_columns=self.numeric_columns, fitted_scaler=self.fitted_scaler )
    return pd.DataFrame(self.model.predict_proba(input_processed)[:,1], columns=['predicted'])

  
  def _preprocess_data(self,
                      X,
                      numeric_columns,
                      fitted_scaler):
    res_df = preprocess_data(X, numeric_columns, fitted_scaler)
    self._df = res_df
    
    return res_df

# COMMAND ----------

# DBTITLE 1,Create XGBoost Classifier Model Fit Method - Return Probability and XGB Model
# Our fit method will be used within our MLflow model training experiment run
# The AUROC metric is chosen here 
def fit(X, y):
  """
   :return: dict with fields 'loss' (scalar loss) and 'model' fitted model instance
  """
  import xgboost
  from xgboost import XGBClassifier
  from sklearn.model_selection import cross_val_score
  
  _model =  XGBClassifier(learning_rate=0.3,
                          gamma=5,
                          max_depth=8,
                          n_estimators=15,
                          min_child_weight = 9, objective='binary:logistic')

  xgb_model = _model.fit(X, y)
  
  score = -cross_val_score(_model, X, y, scoring='roc_auc').mean()
  
  return {'loss': score, 'model': xgb_model}

# COMMAND ----------

# MAGIC %md 
# MAGIC Our input dataset has several fields which will be used for rule based modeling and machine learning. In this notebook we will rely on our machine learning model to identify important features that are effective at predicting fraud. Let's take a look into descriptions of these features to understand our downstream modeling and interpretability results.
# MAGIC <br>
# MAGIC <br>
# MAGIC * LAST_ADR_CHNG_DUR     - Duration in days since the last address change on the account.
# MAGIC <br>
# MAGIC * AVG_DLY_AUTHZN_AMT    - The average daily authorization amount on the plastic since the day of first use.
# MAGIC <br>
# MAGIC * DISTANCE_FROM_HOME	  - Approximate distance of customer's home from merchant.
# MAGIC <br>
# MAGIC * HOME_PHN_NUM_CHNG_DUR - Duration in days since the home phone number was changed on the account.

# COMMAND ----------

# DBTITLE 1,Read Delta Lake for Transactions
from pyspark.sql.functions import * 

import cloudpickle
import pandas as pd
import numpy as np


df = spark.read.format("delta") \
  .load(raw_data_path)

data = df.toPandas()
data = data.drop(columns=['AUTH_ID', 'ACCT_ID_TOKEN'])
numeric_columns = data.columns.to_list()
numeric_columns.remove('FRD_IND')
data.head()

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://brysmiwasb.blob.core.windows.net/demos/dff/dff_data.png">

# COMMAND ----------

# DBTITLE 1,Add xgboost and sklearn to be used in the Docker environment for serving later on
conda_env = mlflow.pyfunc.get_default_conda_env()
conda_env['dependencies'][2]['pip'] += [f'xgboost=={xgb.__version__}']
conda_env['dependencies'][2]['pip'] += [f'scikit-learn=={sklearn.__version__}']

# COMMAND ----------

# DBTITLE 1,MLFlow Tracking and PyFunc Model Saving
import mlflow
useremail = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
experiment_name = f"/Users/{useremail}/dff_orchestrator"
mlflow.set_experiment(experiment_name) 
model_run_name = 'fraud-xgb-wrapper'

with mlflow.start_run() as run:
  mlflow.log_param('Input-data-location', raw_data_path)
  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(data.drop(["FRD_IND"], axis=1), data["FRD_IND"], test_size=0.33, random_state=42)

  from sklearn.preprocessing import StandardScaler 
  # create a scaler for our numeric variables
  # only run this on the training dataset and use to scale test set later.
  scaler = StandardScaler()
  fitted_scaler = scaler.fit(X_train[numeric_columns])
  X_train_processed = preprocess_data(source_df=X_train, numeric_columns=numeric_columns, fitted_scaler=fitted_scaler )

  #train a model and get the loss
  train_dict = {}
  train_dict = fit(X=X_train_processed, y=y_train)
  xgb_model = train_dict['model']
  mlflow.log_metric('loss', train_dict['loss'])
  
  ##------- log pyfunc custom model -------##
   # make an instance of the Pyfunc Class
  myXGB = XGBWrapper(model = xgb_model,
                     X = data[numeric_columns].copy(), 
                     y = data['FRD_IND'], 
                     numeric_columns = numeric_columns)
  
  mlflow.pyfunc.log_model(model_run_name, python_model=myXGB, conda_env=conda_env)

  mlflow.log_metric('auroc', myXGB.auc)
  
# programmatically get the latest Run ID
runs = mlflow.search_runs(mlflow.get_experiment_by_name(experiment_name).experiment_id)
latest_run_id = runs.sort_values('end_time').iloc[-1]["run_id"]
print('The latest run id: ', latest_run_id)

# COMMAND ----------

X = data[numeric_columns].copy()
y = data['FRD_IND']
train_dict = fit(X=X, y=y)
xgb_model = train_dict['model']

predictions = myXGB.predict(spark, X)
predictions.head()

# COMMAND ----------

# DBTITLE 1,Register Model
client = mlflow.tracking.MlflowClient()
model_uri = "runs:/{}/{}".format(latest_run_id, model_run_name)
model_name = "fraud_xgb_model"
result = mlflow.register_model(model_uri, model_name)
version = result.version

# COMMAND ----------

# DBTITLE 1,Transition the model to Production
# archive any production versions of the model from prior runs
for mv in client.search_model_versions("name='{0}'".format(model_name)):
  
    # if model with this name is marked staging
    if mv.current_stage.lower() == 'production':
      # mark is as archived
      client.transition_model_version_stage(
        name=model_name,
        version=mv.version,
        stage='archived'
        )

client.transition_model_version_stage(
  name=model_name,
  version=version,
  stage="Production",
)

# COMMAND ----------

# MAGIC %md 
# MAGIC After running SHAP on model we can see how some of the features such  duration since address change, transaction amount and available cash in the account were proved to be most important. While this is purely machine learning driven approach, we will look at ways to improve customer satisfaction with rule based modeling based vs really totally on ML based approach.

# COMMAND ----------

# DBTITLE 1,Use SHAP for Model explainability
import shap
from pyspark.sql import *
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X, y=y.values)
mean_abs_shap = np.absolute(shap_values).mean(axis=0).tolist()
display(spark.createDataFrame(sorted(list(zip(mean_abs_shap, X.columns)), reverse=True)[:8], ["Mean |SHAP|", "Column"]))

# COMMAND ----------

shap_values = explainer.shap_values(X, y=y.values)
print(shap_values.shape)

# COMMAND ----------

display(shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:],matplotlib=True))

# COMMAND ----------

import pandas as pd 
schema = spark.createDataFrame(X).schema
df = spark.createDataFrame(pd.DataFrame(shap_values, columns=X.columns)).withColumn("id", monotonically_increasing_id())
for col in df.columns:
  df = df.withColumnRenamed(col, 'shap_v_' + col)
df.createOrReplaceTempView("fraud_shap_values")

# COMMAND ----------

spark.createDataFrame(pd.concat([pd.DataFrame(X, columns=X.columns), pd.DataFrame(predictions, columns=['predicted']), pd.DataFrame(y, columns=['FRD_IND'])], axis=1)).withColumn("id", monotonically_increasing_id()).createOrReplaceTempView("txns")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Model Result Saving 
# MAGIC 
# MAGIC In addition to saving model fraud scores, we want to be able to interactively query SHAP values on each observation also. We will persist these values on each observation so we can query in tabular form using SQL Analytics.

# COMMAND ----------

spark.sql("""drop table if exists silver_fraud_shap_values""")
spark.sql("""select t.*, 
       s.*
from txns t join fraud_shap_values s 
on t.id = s.shap_v_id""").write.format("delta").option('overwriteSchema', 'true').mode('overwrite').saveAsTable("silver_fraud_shap_values")

# COMMAND ----------

# DBTITLE 1,Fraud Absolute Dollar Amounts - Predicted vs Actual Amount Lost
# MAGIC %sql 
# MAGIC 
# MAGIC select case when predicted > 0.5 then 1 else 0 end predicted_Ind, frd_ind, count(1) ct
# MAGIC from silver_fraud_shap_values
# MAGIC group by case when predicted > 0.5 then 1 else 0 end, frd_ind

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC + <a href="$./01_dff_model">STAGE1</a>: Integrating rule based with ML
# MAGIC + <a href="$./02_dff_orchestration">STAGE2</a>: Building a fraud detection model
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License [https://databricks.com/db-license-source].  All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC 
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | shap                                   | Model explainability    | MIT        | https://github.com/slundberg/shap                   |
# MAGIC | networkx                               | Graph toolkit           | BSD        | https://github.com/networkx                         |
# MAGIC | xgboost                                | Gradient Boosting lib.  | Apache2    | https://github.com/dmlc/xgboost                     |
# MAGIC | graphviz                               | Network visualization   | MIT        | https://github.com/xflr6/graphviz                   |
# MAGIC | pandasql                               | SQL syntax on pandas    | MIT  | https://github.com/yhat/pandasql/                   |
# MAGIC | pydot                                  | Network visualization   | MIT        | https://github.com/pydot/pydot                      |
# MAGIC | pygraphviz                             | Network visualization   | BSD        | https://pygraphviz.github.io/                       |

# COMMAND ----------


