# Databricks notebook source
# MAGIC %md 
# MAGIC You may find this series of notebooks at https://github.com/databricks-industry-solutions/fraud-orchestration. For more information about this solution accelerator, visit https://www.databricks.com/solutions/accelerators/fraud-detection.

# COMMAND ----------

# MAGIC %md
# MAGIC <img src=https://brysmiwasb.blob.core.windows.net/demos/dff/databricks_fsi_white.png width="600px">

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Databricks fraud framework - Orchestration
# MAGIC 
# MAGIC The financial service industry (FSI) is rushing towards transformational change to support new channels and services, delivering transactional features and facilitating payments through new digital channels to remain competitive. Unfortunately, the speed and convenience that these capabilities afford is a benefit to consumers and fraudsters alike. Building a fraud framework often goes beyond just creating a highly accurate machine learning model due ever changing landscape and customer expectation. Oftentimes it involves a complex decision science setup which combines rules engine with a need for a robust and scalable machine learning platform. In this series of notebook, we'll be demonstrating how `Delta Lake`, `MLFlow` and a unified analytics platform can help organisations combat fraud more efficiently
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

# DBTITLE 1,Install binaries for graphviz
# MAGIC %sh -e sudo apt-get install graphviz libgraphviz-dev pkg-config -y

# COMMAND ----------

# MAGIC %pip install networkx==2.4 pandasql==0.7.3 graphviz==0.16 sqlalchemy==1.4.46 pygraphviz==1.7 pydot==1.4.2

# COMMAND ----------

# DBTITLE 1,Extract rules from DMN file
# MAGIC %sh
# MAGIC cat ./DFF_Ruleset.dmn

# COMMAND ----------

# DBTITLE 1,Build graph from DMN format
from xml.dom import minidom
import pandas as pd
import networkx as nx
import xgboost
import sklearn

xmldoc = minidom.parse('./DFF_Ruleset.dmn')
itemlist = xmldoc.getElementsByTagName('dmn:decision')

G = nx.DiGraph()
for item in itemlist:
  
    node_id = item.attributes['id'].value
    node_decision = str(item.attributes['name'].value)
    G.add_node(node_id, decision=node_decision)
    
    infolist = item.getElementsByTagName("dmn:informationRequirement")
    if(len(infolist) > 0):
      info = infolist[0]
      for req in info.getElementsByTagName("dmn:requiredDecision"):
        parent_id = req.attributes['href'].value.split('#')[-1]
        G.add_edge(parent_id, node_id)

# COMMAND ----------

# DBTITLE 1,Render graph
from graphviz import Digraph

filename = '/tmp/dff_model'
extension = 'svg'

def toGraphViz(g):
  dot = Digraph(comment='The Fraud Engine', format=extension, filename=filename)
  atts = nx.get_node_attributes(G, 'decision')
  for node in atts:
    att = atts[node]
    dot.node(node, att, color='black', shape='box', fontname="courier")
  for edge in g.edges:
    dot.edge(edge[0], edge[1])
  return dot

dot = toGraphViz(G)
dot.render()
displayHTML(dot.pipe().decode('utf-8'))

# COMMAND ----------

# DBTITLE 1,Validate DAG
if (not nx.is_directed_acyclic_graph(G)):
  raise Exception('Workflow is not a DAG')

# COMMAND ----------

# DBTITLE 1,Topological sorting
# Our core logic is to traverse our graph in order, calling parent rules before children
# Although we could recursively parse our tree given a root node ID, it is much more convenient (and less prone to error) to sort our graph topologically
# ... accessing each rule in each layer
decisions = nx.get_node_attributes(G, 'decision')
pd.DataFrame([decisions[rule] for rule in nx.topological_sort(G)], columns=['stage'])

# COMMAND ----------

# DBTITLE 1,Create our orchestrator model
import mlflow.pyfunc

class DFF_Model(mlflow.pyfunc.PythonModel):
  
  import networkx as nx
  import pandas as pd
  from pandasql import sqldf
  
  '''
  For rule based, we simply match record against predefined SQL where clause
  If rule matches, we return 1, else 0
  '''
  def func_sql(self, sql):
    from pandasql import sqldf
    # We do not execute the match yet, we simply return a function that will be called later
    # This allow us to define function only once
    # TODO: Prevent model SQL injections :)
    def func_sql2(input_df):
      pred = sqldf("SELECT CASE WHEN {} THEN 1 ELSE 0 END AS predicted FROM input_df".format(sql)).predicted.iloc[0]
      return pred
    return func_sql2
  
  '''
  For model based, we execute model against record
  We return model prediction (between 0 and 1)
  '''
  def func_model(self, uri):
    model = mlflow.pyfunc.load_model(uri)
    # We do not execute the match yet, we simply return a function that will be called later
    # This allow us to load a model from MLFlow only once
    def func_model2(df):
      pred_df = model.predict(df).predicted
      return pred_df.iloc[0]
    return func_model2
  
  '''
  We define our PyFunc model using a DAG (a serialized NetworkX object) and a predefined sensitivity
  Although rule based would be binary (0 or 1), ML based would not necessarily, and we need to define a sensitivity upfront 
  to know if we need to traverse our tree any deeper (in case we chain multiple ML models)
  '''
  def __init__(self, G, sensitivity):
    self.G = G
    self.sensitivity = sensitivity
  
  '''
  At model startup, we traverse our DAG and load all business logic required at scoring phase
  Although it does not change much on the rule execution logic, we would be loading models only once at model startup (not at scoring)
  '''
  def load_context(self, context):
    
    rules = []
    decisions = nx.get_node_attributes(self.G, 'decision')

    # topological sort will explore each node in the right order, exploring parents before children nodes
    for i, rule in enumerate(nx.topological_sort(self.G)):
      
      # we retrieve the SQL syntax of the rule or the URI of a model
      decision = decisions[rule]
      if(decision.startswith("models:/")):
        # we load ML model only once as a function that we can call later
        rules.append([i, decision, self.func_model(decision)])
      else:
        # we load a SQL statement as a function that we can call later
        rules.append([i, decision, self.func_sql(decision)])
      
    self.rs = rules
  
  
  def predict_record(self, s):
    
    model_input = pd.DataFrame([s.values], columns=s.index)
    for i, rule, F in self.rs:
      # run next rule on 
      pred = F(model_input)
      
      # rule / model matches, return fraudulent record
      if(pred >= self.sensitivity):
        return rule
        
    return None
  
  '''
  After multiple considerations, we defined our model to operate on a single record only and not against an entire dataframe
  This helps us to be much more precise in what data was triggered against what rule / model and what chunk would need to be 
  evaluated further
  '''
  def predict(self, context, df):
    return df.apply(self.predict_record, axis=1)

# COMMAND ----------

# DBTITLE 1,Include 3rd party dependencies
# we may have to store additional libraries such as networkx and pandasql
conda_env = mlflow.pyfunc.get_default_conda_env()
conda_env['dependencies'][2]['pip'] += [f'networkx==2.4']
conda_env['dependencies'][2]['pip'] += [f'pandasql==0.7.3']
conda_env['dependencies'][2]['pip'] += [f'xgboost=={xgboost.__version__}']
conda_env['dependencies'][2]['pip'] += [f'scikit-learn=={sklearn.__version__}']
conda_env

# COMMAND ----------

# DBTITLE 1,Create our experiment
useremail = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
experiment_name = f"/Users/{useremail}/dff_orchestrator"
mlflow.set_experiment(experiment_name) 
with mlflow.start_run(run_name='fraud_model'):
  # we define a sensitivity of 0.7, that is that probability of a record to be fraudulent for ML model needs to be at least 70%
  # TODO: explain how sensitivity could be dynamically pulled from a MLFlow model (tag, metrics, etc.)
  mlflow.pyfunc.log_model('model', python_model=DFF_Model(G, 0.7), conda_env=conda_env)
  mlflow.log_artifact("{}.{}".format(filename, extension))
  run_id = mlflow.active_run().info.run_id

# COMMAND ----------

# DBTITLE 1,Register framework
client = mlflow.tracking.MlflowClient()
model_uri = "runs:/{}/model".format(run_id)
model_name = "dff_orchestrator"
result = mlflow.register_model(model_uri, model_name)
version = result.version

# COMMAND ----------

# DBTITLE 1,Register model to staging
# archive any staging versions of the model from prior runs
for mv in client.search_model_versions("name='{0}'".format(model_name)):
  
    # if model with this name is marked staging
    if mv.current_stage.lower() == 'staging':
      # mark is as archived
      client.transition_model_version_stage(
        name=model_name,
        version=mv.version,
        stage='archived'
        )
      
client.transition_model_version_stage(
  name=model_name,
  version=version,
  stage="Staging",
)

# COMMAND ----------

# DBTITLE 1,Create widgets
dbutils.widgets.text("CDHLDR_PRES_CD", "0")
dbutils.widgets.text("ACCT_CL_AMT", "10000")
dbutils.widgets.text("LAST_ADR_CHNG_DUR", "301")
dbutils.widgets.text("ACCT_AVL_CASH_BEFORE_AMT", "100")
dbutils.widgets.text("AUTHZN_AMT", "30")
dbutils.widgets.text("DISTANCE_FROM_HOME", "1000")
dbutils.widgets.text("AUTHZN_OUTSTD_CASH_AMT", "40")
dbutils.widgets.text("AVG_DLY_AUTHZN_AMT", "25")

# COMMAND ----------

#run_id
# Score dataframe against DFF orchestration engine
model = mlflow.pyfunc.load_model("runs:/{}/model".format(run_id))

# COMMAND ----------

# DBTITLE 1,Validate framework
import random
from graphviz import Digraph

df_dict = {}
for col in ['ACCT_PROD_CD', 'ACCT_AVL_CASH_BEFORE_AMT', 'ACCT_AVL_MONEY_BEFORE_AMT',
       'ACCT_CL_AMT', 'ACCT_CURR_BAL', 'APPRD_AUTHZN_CNT',
       'APPRD_CASH_AUTHZN_CNT', 'AUTHZN_AMT', 'AUTHZN_OUTSTD_AMT',
       'AVG_DLY_AUTHZN_AMT', 'AUTHZN_OUTSTD_CASH_AMT', 'CDHLDR_PRES_CD',
       'HOTEL_STAY_CAR_RENTL_DUR', 'LAST_ADR_CHNG_DUR',
       'HOME_PHN_NUM_CHNG_DUR', 'PLSTC_ISU_DUR', 'POS_COND_CD',
       'POS_ENTRY_MTHD_CD', 'DISTANCE_FROM_HOME', 'FRD_IND']:
  try:
    df_dict[col] = [float(dbutils.widgets.get(col))]
  except:
    df_dict[col] = [random.uniform(1, 10)]

pdf = pd.DataFrame.from_dict(df_dict)

# Score dataframe against DFF orchestration engine
model = mlflow.pyfunc.load_model("runs:/{}/model".format(run_id))
decision = model.predict(pdf).iloc[0]

# Visualize our rule set and which one was triggered (if any)
def toGraphViz_triggered(g):
  dot = Digraph(comment='The Fraud Engine', format='svg', filename='/tmp/dff_triggered')
  atts = nx.get_node_attributes(G, 'decision')
  for node in atts:
    att = atts[node]
    if(att == decision):
      dot.node(node, att, color='red', shape='box', fontname="courier")
    else:
      dot.node(node, att, color='black', shape='box', fontname="courier")
  for edge in g.edges:
    dot.edge(edge[0], edge[1])
  return dot

dot = toGraphViz_triggered(G)
dot.render()
displayHTML(dot.pipe().decode('utf-8'))

# COMMAND ----------

# DBTITLE 1,Model Serving Test (Enable Model Serving to run)
import os
import requests
import pandas as pd

import statistics

def score_model(dataset: pd.DataFrame):
  token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
  url = 'https://e2-demo-field-eng.cloud.databricks.com/model/{0}/Staging/invocations'.format(model_name) # update to the url of your own workspace
  headers = {'Authorization': f'Bearer {token}'}
  data_json = {"dataframe_split": dataset.to_dict(orient='split')}
  response = requests.request(method='POST', headers=headers, url=url, json=data_json)
  if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')
  return response.json()

try:
  decision = score_model(pdf)['predictions'][0]['0']
  if (decision is None ):
    displayHTML("<h3>VALID TRANSACTION</h3>")
  else:
    displayHTML("<h3>FRAUDULENT TRANSACTION: {}</h3>".format(decision))
except:
  displayHTML("<h3>ENABLE MODEL SERVING</h3>")

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
# MAGIC | pandasql                               | SQL syntax on pandas    | Yhat, Inc  | https://github.com/yhat/pandasql/                   |
# MAGIC | pydot                                  | Network visualization   | MIT        | https://github.com/pydot/pydot                      |
# MAGIC | pygraphviz                             | Network visualization   | BSD        | https://pygraphviz.github.io/                       |

# COMMAND ----------


