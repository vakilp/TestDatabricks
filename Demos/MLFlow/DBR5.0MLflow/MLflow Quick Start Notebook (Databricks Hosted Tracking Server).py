# Databricks notebook source
# MAGIC %md ## MLflow Quick Start Notebook (Databricks Hosted Tracking Server)
# MAGIC This is a Quick Start notebook based on [MLflow's tutorial](https://mlflow.org/docs/latest/tutorial.html).  In this tutorial, we’ll:
# MAGIC * Install the MLflow library on a Databricks cluster
# MAGIC * Connect our notebook to an MLflow Tracking Server that is hosted by Databricks
# MAGIC * Log metrics, parameters, models and a .png plot to show how you can record arbitrary outputs from your MLflow job
# MAGIC * View our results on the MLflow tracking UI.
# MAGIC 
# MAGIC This notebook uses the `diabetes` dataset in scikit-learn and predicts the progression metric (a quantitative measure of disease progression after one year after) based on BMI, blood pressure, etc. It uses the scikit-learn ElasticNet linear regression model, where we vary the `alpha` and `l1_ratio` parameters for tuning. For more information on ElasticNet, refer to:
# MAGIC   * [Elastic net regularization](https://en.wikipedia.org/wiki/Elastic_net_regularization)
# MAGIC   * [Regularization and Variable Selection via the Elastic Net](https://web.stanford.edu/~hastie/TALKS/enet_talk.pdf)
# MAGIC 
# MAGIC A good reference for MLflow in general is [Matei's Spark Summit 2018 Keynote](https://databricks.com/sparkaisummit/north-america/spark-summit-2018-keynotes).
# MAGIC 
# MAGIC To get started, you will first need to 
# MAGIC 
# MAGIC 1. Be part of the Databricks Hosted MLflow early adopter program and have MLflow tracking server enabled on your shard.
# MAGIC 2. Install the most recent version of MLflow and Python ML and math libraries on your Databricks cluster (see details in the next cell).
# MAGIC 
# MAGIC This notebook is a modified version of the [this notebook published in the Databricks docs](https://docs.databricks.com/spark/latest/mllib/mlflow.html#mlflow-quick-start-model-training-and-logging) which itself is a version of the [MLflow Quick Start Notebook](https://docs.databricks.com/_static/notebooks/mlflow/mlflow-quick-start-notebook.html) in [Databricks Documentation](https://docs.databricks.com/)

# COMMAND ----------

# MAGIC %md ### Install MLflow on Your Databricks Cluster
# MAGIC 
# MAGIC ####NOTE: ML Runtime 5.0 beta or newer is required!
# MAGIC  
# MAGIC  
# MAGIC 1. Ensure you are using or [create a cluster](https://docs.databricks.com/user-guide/clusters/create.html) specifying 
# MAGIC   * **Databricks Runtime Version:** Databricks Runtime "5.0 beta" 
# MAGIC   * **Python Version:** Python 3
# MAGIC 2. Install `mlflow` as a [PiPy library](https://docs.databricks.com/user-guide/libraries.html#upload-a-python-pypi-package-or-python-egg).
# MAGIC   1. Choose **PyPi** and enter `mlflow==0.6.0`.
# MAGIC 3. For our ElasticNet Descent Path visualizations, install the latest `scikit-learn` and `matplotlib` as a PyPI libraries.
# MAGIC   1. Choose **PyPi** and enter `scikit-learn==0.19.1`
# MAGIC   1. Choose **PyPi** and enter `matplotlib==2.2.2`

# COMMAND ----------

# MAGIC %md #### Write Your ML Code Based on the`train_diabetes.py` Code
# MAGIC This tutorial is based on the MLflow's [train_diabetes.py](https://github.com/databricks/mlflow/blob/master/example/tutorial/train_diabetes.py), which uses the `sklearn.diabetes` built-in dataset to predict disease progression based on various factors.

# COMMAND ----------

# Import various libraries including matplotlib, sklearn, mlflow
import os
import warnings
import sys

import pandas as pd
import numpy as np
from itertools import cycle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import lasso_path, enet_path
from sklearn import datasets

# Import mlflow
import mlflow
import mlflow.sklearn

# Load Diabetes datasets
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

# Create pandas DataFrame for sklearn ElasticNet linear_model
Y = np.array([y]).transpose()
d = np.concatenate((X, Y), axis=1)
cols = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6', 'progression']
data = pd.DataFrame(d, columns=cols)

# COMMAND ----------

# MAGIC %md #### Plot the ElasticNet Descent Path
# MAGIC As an example of recording arbitrary output files in MLflow, we'll plot the [ElasticNet Descent Path](http://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_coordinate_descent_path.html) for the ElasticNet model by *alpha* for the specified *l1_ratio*.
# MAGIC 
# MAGIC The `plot_enet_descent_path` function below:
# MAGIC * Returns an image that can be displayed in our Databricks notebook via `display`
# MAGIC * As well as saves the figure `ElasticNet-paths.png` to the Databricks cluster's driver node
# MAGIC * This file is then uploaded to MLflow using the `log_artifact` within `train_diabetes`

# COMMAND ----------

def plot_enet_descent_path(X, y, l1_ratio):
    # Compute paths
    eps = 5e-3  # the smaller it is the longer is the path

    # Reference the global image variable
    global image
    
    print("Computing regularization path using the elastic net.")
    alphas_enet, coefs_enet, _ = enet_path(X, y, eps=eps, l1_ratio=l1_ratio, fit_intercept=False)

    # Display results
    fig = plt.figure(1)
    ax = plt.gca()

    colors = cycle(['b', 'r', 'g', 'c', 'k'])
    neg_log_alphas_enet = -np.log10(alphas_enet)
    for coef_e, c in zip(coefs_enet, colors):
        l1 = plt.plot(neg_log_alphas_enet, coef_e, linestyle='--', c=c)

    plt.xlabel('-Log(alpha)')
    plt.ylabel('coefficients')
    title = 'ElasticNet Path by alpha for l1_ratio = ' + str(l1_ratio)
    plt.title(title)
    plt.axis('tight')

    # Display images
    image = fig
    
    # Save figure
    fig.savefig("ElasticNet-paths.png")

    # Close plot
    plt.close(fig)

    # Return images
    return image    

# COMMAND ----------

# MAGIC %md #### Train the Diabetes Model
# MAGIC The next function trains Elastic-Net linear regression based on the input parameters of `alpha (in_alpha)` and `l1_ratio (in_l1_ratio)`.
# MAGIC 
# MAGIC In addition, this function uses MLflow Tracking to record its
# MAGIC * parameters
# MAGIC * metrics
# MAGIC * model
# MAGIC * arbitrary files, namely the above noted Lasso Descent Path plot.
# MAGIC 
# MAGIC **Tip:** We use `with mlflow.start_run:` in the Python code to create a new MLflow run. This is the recommended way to use MLflow in notebook cells. Whether your code completes or exits with an error, the `with` context will make sure that we close the MLflow run, so you don't have to call `mlflow.end_run` later in the code.

# COMMAND ----------

myMLflow = mlflow.tracking.MlflowClient(tracking_uri="databricks")
expInfo = myMLflow.get_experiment(20)
expInfo
#expID = myMLflow.get_experiment_by_name('quickstartPSV')
#expID
expID = 20

# COMMAND ----------

# train_diabetes
#   Uses the sklearn Diabetes dataset to predict diabetes progression using ElasticNet
#       The predicted "progression" column is a quantitative measure of disease progression one year after baseline
#       http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html
#
#   Returns: The MLflow RunInfo associated with this training run, see
#            https://mlflow.org/docs/latest/python_api/mlflow.entities.html#mlflow.entities.RunInfo
#            We will use this later in the notebook to demonstrate ways to access the output of this
#            run and do useful things with it!
def train_diabetes(data, in_alpha, in_l1_ratio):
  # Evaluate metrics
  def eval_metrics(actual, pred):
      rmse = np.sqrt(mean_squared_error(actual, pred))
      mae = mean_absolute_error(actual, pred)
      r2 = r2_score(actual, pred)
      return rmse, mae, r2

  warnings.filterwarnings("ignore")
  np.random.seed(40)

  # Split the data into training and test sets. (0.75, 0.25) split.
  train, test = train_test_split(data)

  # The predicted column is "progression" which is a quantitative measure of disease progression one year after baseline
  train_x = train.drop(["progression"], axis=1)
  test_x = test.drop(["progression"], axis=1)
  train_y = train[["progression"]]
  test_y = test[["progression"]]

  if float(in_alpha) is None:
    alpha = 0.05
  else:
    alpha = float(in_alpha)
    
  if float(in_l1_ratio) is None:
    l1_ratio = 0.05
  else:
    l1_ratio = float(in_l1_ratio)
  
  # Start an MLflow run; the "with" keyword ensures we'll close the run even if this cell crashes
  with mlflow.start_run(experiment_id=expID) as run:
    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr.fit(train_x, train_y)

    predicted_qualities = lr.predict(test_x)

    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    # Print out ElasticNet model metrics
    print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

    # Set tracking_URI first and then reset it back to not specifying port
    # Note, we had specified this in an earlier cell
    #mlflow.set_tracking_uri(mlflow_tracking_URI)

    # Log mlflow attributes for mlflow UI
    mlflow.log_param("alpha", alpha)
    mlflow.log_param("l1_ratio", l1_ratio)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)
    mlflow.sklearn.log_model(lr, "model")
    
    # Call plot_enet_descent_path
    image = plot_enet_descent_path(X, y, l1_ratio)
    
    # Log artifacts (output files)
    mlflow.log_artifact("ElasticNet-paths.png")
    
    print("Inside MLflow Run with id %s" % run.info.run_uuid)
    
    # return our RunUUID so we can use it when we try out some other APIs later in this notebook.
    return run.info

# COMMAND ----------

# MAGIC %md
# MAGIC ![](https://docs.databricks.com/_static/images/mlflow/elasticnet-paths-by-alpha-per-l1-ratio.png)

# COMMAND ----------

display(myMLflow.list_experiments())

# COMMAND ----------

runInfo = myMLflow.list_run_infos(expInfo.experiment_id)
display(runInfo)

# COMMAND ----------

myMLflow.list_artifacts(runInfo[0].run_uuid)

# COMMAND ----------

# MAGIC %md #### Experiment with Different Parameters
# MAGIC 
# MAGIC Now that we have a `train_diabetes` function that records MLflow runs, we can simply call it with different parameters to explore them. Later, we'll be able to visualize all these runs on our MLflow tracking server.

# COMMAND ----------

# Start with alpha and l1_ratio values of 0.01, 0.01
run_info_1 = train_diabetes(data, 0.01, 0.01)

# COMMAND ----------

display(image)

# COMMAND ----------

# Start with alpha and l1_ratio values of 0.01, 0.75
run_info_2 = train_diabetes(data, 0.01, 0.75)

# COMMAND ----------

display(image)

# COMMAND ----------

# Start with alpha and l1_ratio values of 0.01, 1
run_info_3 = train_diabetes(data, 0.01, 1)

# COMMAND ----------

display(image)

# COMMAND ----------

# MAGIC %md ## Review the MLflow UI
# MAGIC Visit your tracking server in a web browser by going to `https://your_shard_id.cloud.databricks.com/mlflow`

# COMMAND ----------

# MAGIC %md
# MAGIC The MLflow UI should look something similar to the animated GIF below. Inside the UI, you can:
# MAGIC * View your experiments and runs
# MAGIC * Review the parameters and metrics on each run
# MAGIC * Click each run for a detailed view to see the the model, images, and other artifacts produced.
# MAGIC 
# MAGIC <img src="https://docs.databricks.com/_static/images/mlflow/mlflow-ui.gif"/>

# COMMAND ----------

# MAGIC %md #### SIDE BAR: Organize MLflow Runs into Experiments
# MAGIC 
# MAGIC As you start using your MLflow server for more tasks, you may want to separate them out. MLflow allows you to create [experiments](https://mlflow.org/docs/latest/tracking.html#organizing-runs-in-experiments) to organize your runs. To report your run to a specific experiment, just pass an `experiment_id` parameter to the `mlflow.start_run`, as in `mlflow.start_run(experiment_id=1)`.
# MAGIC 
# MAGIC Note that the experiments we ran above did not specify an `experiment_id` parameter so they defaulted to the "Default Experiment" which has ID 0.

# COMMAND ----------

# MAGIC %md ## Load MLflow model back as a Scikit-learn model
# MAGIC Here we demonstrate using the MLflow API to load model from the MLflow server that was produced by a given run. To do so we have to specify the run_id.
# MAGIC 
# MAGIC Once we load it back in, it is a just a scikitlearn model object like any other and we can explore it or use it.

# COMMAND ----------

import mlflow.sklearn
model = mlflow.sklearn.load_model(path="model", run_id=run_info_1.run_uuid) #Use one of the run IDs we captured above
model.coef_

# COMMAND ----------

#Get a prediction for a row of the dataset
model.predict(data[0:1].drop(["progression"], axis=1))

# COMMAND ----------

# MAGIC %md ## Use an MLflow Model for Batch inference
# MAGIC We can also get a pyspark UDF to do some batch inference suing one of the models you logged above. For more on this see https://mlflow.org/docs/latest/models.html#apache-spark

# COMMAND ----------

# First let's create a Spark DataFrame out of our original pandas
# DataFrame minus the column we want to predict. We'll use this
# to simulate what this would be like if we had a big data set
# that was regularly getting updated that we were routinely wanting
# to score, e.g. click logs.
df = spark.createDataFrame(data.drop(["progression"], axis=1))

# COMMAND ----------

# Next we use the MLflow API to create a PySpark UDF given our run.
# See the API docs for this function call here:
# https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.spark_udf
# the spark_udf function takes our SparkSession, the path to the model within artifact
# repository, and the ID of the run that produced this model.
pyfunc_udf = mlflow.pyfunc.spark_udf(spark, "model", run_id=run_info_1.run_uuid)

# COMMAND ----------

#withColumns adds a column to the data by applying the python UDF to the DataFrame
predicted_df = df.withColumn("prediction", pyfunc_udf(
  'age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6'))
display(predicted_df)

# COMMAND ----------

# MAGIC %fs ls /FileStore/plots

# COMMAND ----------

