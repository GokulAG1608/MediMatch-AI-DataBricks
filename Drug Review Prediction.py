# Databricks notebook source
# MAGIC %md
# MAGIC INSATLLING NECESSARY LIBRARIES

# COMMAND ----------

pip install mlflow

# COMMAND ----------

pip install Pipeline

# COMMAND ----------

# MAGIC %md
# MAGIC Feteching the Datas from the DBFS 

# COMMAND ----------

# MAGIC %fs ls /FileStore/shared_uploads/aggokul2000@gmail.com

# COMMAND ----------

# MAGIC %md
# MAGIC Importing the Necessary Libraries
# MAGIC

# COMMAND ----------

import mlflow
import mlflow.spark
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, Tokenizer, HashingTF, IDF, StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.sql.functions import when, col
from pyspark.sql.types import *
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession

# Set MLflow experiment
mlflow.set_experiment("/Shared/Pharma_Logistic_Regression")

# Initialize MLflow
mlflow.spark.autolog()

# COMMAND ----------

# MAGIC %md
# MAGIC Ingestion the Data into DataBricks

# COMMAND ----------

spark = SparkSession.builder.appName("Drug Review Prediction Project ").getOrCreate()

# COMMAND ----------

data = spark.read.format("csv").option("header",True).option("inferschema",True).load("dbfs:/FileStore/shared_uploads/aggokul2000@gmail.com/drug.csv")

schema = StructType([
    StructField("uniqueID", StringType(), True),
    StructField("drugName", StringType(), True),
    StructField("condition", StringType(), True),
    StructField("review", StringType(), True),
    StructField("rating", IntegerType(), True),
    StructField("date", StringType(), True),
    StructField("usefulCount", IntegerType(), True)
])

df = spark.read.format("csv").option("header",True).option("inferschema",True).option("multiline", "true").option("escape", "\"").option("quote", "\"").load("dbfs:/FileStore/shared_uploads/aggokul2000@gmail.com/drug.csv",schema = schema)

# COMMAND ----------

# MAGIC %md
# MAGIC PreProcessing and Transformation

# COMMAND ----------

# Data Cleaning and Feature Engineering
df = df.na.fill({"uniqueID": 0, "drugName": "unknown","condition":"unknown","review":"unknown","date":0,"usefulCount":0})

df = df.filter(col("rating").isNotNull()).filter(col("rating").cast("int").isNotNull())

# Feature Transformation Pipeline
# Define feature transformation stages
indexer_drug = StringIndexer(inputCol="drugName", outputCol="drugName_index",handleInvalid="keep")
indexer_condition = StringIndexer(inputCol="condition", outputCol="condition_index",handleInvalid="keep")
encoder_drug = OneHotEncoder(inputCol="drugName_index", outputCol="drugName_vec",handleInvalid="keep")
encoder_condition = OneHotEncoder(inputCol="condition_index", outputCol="condition_vec",handleInvalid="keep")
tokenizer = Tokenizer(inputCol="review", outputCol="review_words")
hashingTF = HashingTF(inputCol="review_words", outputCol="review_tf", numFeatures=100)
idf = IDF(inputCol="review_tf", outputCol="review_tfidf")
assembler = VectorAssembler(inputCols=["drugName_vec", "condition_vec", "review_tfidf", "usefulCount"], outputCol="features")

# Create a pipeline
pipeline = Pipeline(stages=[indexer_drug, indexer_condition, encoder_drug, encoder_condition, tokenizer, hashingTF, idf, assembler])

# COMMAND ----------

# MAGIC %md
# MAGIC Model Evaluation and Prediction

# COMMAND ----------

# MAGIC %md
# MAGIC Logistic Regression

# COMMAND ----------

# Train-test split
train, test = df.randomSplit([0.8, 0.2], seed=42)

# Train pipeline and model
pipeline_model = pipeline.fit(df)
train_transformed = pipeline_model.transform(train)

# Save the trained pipeline
pipeline_path = "/dbfs/FileStore/pipeline/"
pipeline_model.write().overwrite().save(pipeline_path)# Train logistic regression model
lr = LogisticRegression(featuresCol="features", labelCol="rating", maxIter=10, family="multinomial")

with mlflow.start_run(run_name="Pharma Logistic Regression"):
    lr_model = lr.fit(train_transformed)
    
    # Log parameters and metrics
    mlflow.log_param("maxIter", 10)
    predictions = lr_model.transform(pipeline_model.transform(test))
    evaluator = MulticlassClassificationEvaluator(labelCol="rating", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    mlflow.log_metric("Accuracy", accuracy)

    # Log the trained model
    mlflow.spark.log_model(lr_model, "model")
    print(f"Logged Model with Accuracy: {accuracy}")

# COMMAND ----------

from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import mlflow

# Initialize Random Forest Classifier
rf = RandomForestClassifier(featuresCol="features", labelCol="rating", numTrees=50, maxDepth=5)

# Start an MLflow run
with mlflow.start_run(run_name="Pharma Random Forest Classifier"):
    # Train the Random Forest model
    rf_model = rf.fit(train_transformed)
    
    # Log parameters
    mlflow.log_param("numTrees", 50)
    mlflow.log_param("maxDepth", 5)
    
    # Evaluate the model on test data
    predictions = rf_model.transform(pipeline_model.transform(test))
    evaluator = MulticlassClassificationEvaluator(labelCol="rating", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    mlflow.log_metric("Accuracy", accuracy)
    
    # Log the trained model to MLflow
    mlflow.spark.log_model(rf_model, "random_forest_model")
    print(f"Logged Random Forest Model with Accuracy: {accuracy}")

# COMMAND ----------

# MAGIC %md
# MAGIC Decision Tree

# COMMAND ----------

from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import mlflow

# Initialize Decision Tree Classifier
dt = DecisionTreeClassifier(featuresCol="features", labelCol="rating", maxDepth=5)

# Start an MLflow run
with mlflow.start_run(run_name="Pharma Decision Tree Classifier"):
    # Train the Decision Tree model
    dt_model = dt.fit(train_transformed)
    
    # Log parameters
    mlflow.log_param("maxDepth", 5)
    
    # Evaluate the model on test data
    predictions = dt_model.transform(pipeline_model.transform(test))
    evaluator = MulticlassClassificationEvaluator(labelCol="rating", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    mlflow.log_metric("Accuracy", accuracy)
    
    # Log the trained model to MLflow
    mlflow.spark.log_model(dt_model, "decision_tree_model")
    print(f"Logged Decision Tree Model with Accuracy: {accuracy}")


# COMMAND ----------

# MAGIC %md
# MAGIC Creating the Delta Lake Table 

# COMMAND ----------

# MAGIC %sql
# MAGIC create table final_drug using delta location '/hive_metastore.default.final_drug'

# COMMAND ----------

# MAGIC %md
# MAGIC Upload the Final Data into the Delta Table

# COMMAND ----------

simple = predictions.select("condition", "drugName", "rating", "prediction").dropDuplicates()
display(simple)

# COMMAND ----------

simple.write.format("delta").mode("overwrite").option("overwriteschema",True).save("/hive_metastore.default.final_drug")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from final_drug

# COMMAND ----------

# MAGIC %md
# MAGIC New Data Prediction using the MLFLOW 

# COMMAND ----------

# MAGIC %md
# MAGIC New data using the Logistic regression

# COMMAND ----------

# Define schema for the new data
schema = StructType([
    StructField("uniqueID", StringType(), True),
    StructField("drugName", StringType(), True),
    StructField("condition", StringType(), True),
    StructField("review", StringType(), True),
    StructField("rating", IntegerType(), True),
    StructField("date", StringType(), True),
    StructField("usefulCount", IntegerType(), True)
])

# Load new data
new_input_data =  [
    ("1001", "Valsartan", "Left Ventricular Dysfunction", "I feel much better.", 9, "17-May-2018", 20),
    ("1002", "Guanfacine", "ADHD", "Side effects are mild.", 7, "16-Aug-2018", 192),
    ("1003", "Dolo 650", "Body pain", "I feel much better", 8, "03-Aug-2021", 20),
    ("1004", "Ibuprofen", "Headache", "Works great!", 10, "11-Dec-2020", 10),
    ("1005", "Metformin", "Diabetes", "Very effective", 9, "20-Jan-2019", 15)
]
new_data = spark.createDataFrame(new_input_data, schema=schema)

# Preprocess new data (ensure the same preprocessing as the training data)
new_data = new_data.na.fill({"uniqueID": 0, "drugName": "unknown", "condition": "unknown", "review": "unknown", "date": 0, "usefulCount": 0})
new_data = new_data.filter(col("rating").isNotNull()).filter(col("rating").cast("int").isNotNull())

# Load the saved pipeline for feature transformations
pipeline_path = "/dbfs/FileStore/pipeline/"
pipeline_model = PipelineModel.load(pipeline_path)

# Transform the new data
transformed_new_data = pipeline_model.transform(new_data)

# Load the trained logistic regression model from MLflow
logged_model_uri = "runs:/7ee62b18418d4c549481b29eeff3a707/model"  # Replace <run_id> with your actual run ID
lr_model = mlflow.spark.load_model(logged_model_uri)

# Make predictions on new data
predictions = lr_model.transform(transformed_new_data)

# Display predictions
simple = predictions.select( "drugName", "condition", "rating","prediction").dropDuplicates()
display(simple)


# COMMAND ----------

# MAGIC %md
# MAGIC New data Prediction using the Random Forest

# COMMAND ----------

# Define schema for the new data
schema = StructType([
    StructField("uniqueID", StringType(), True),
    StructField("drugName", StringType(), True),
    StructField("condition", StringType(), True),
    StructField("review", StringType(), True),
    StructField("rating", IntegerType(), True),
    StructField("date", StringType(), True),
    StructField("usefulCount", IntegerType(), True)
])

# Load new data
new_input_data = [
    ("1001", "Aspirin", "Hypertension", "It worked well for me.", 9, "01-Jan-2021", 45),
    ("1002", "Lisinopril", "Diabetes", "Had mild side effects.", 7, "15-Feb-2021", 100),
    ("1003", "Metformin", "Heart Disease", "Significantly improved my condition.", 8, "12-Mar-2021", 67),
    ("1004", "Ibuprofen", "Pain", "Didn't work as expected.", 5, "20-Apr-2021", 12),
    ("1005", "Amoxicillin", "Infection", "Highly recommend it.", 10, "10-May-2021", 230),
    ("1006", "Atorvastatin", "Anxiety", "Not effective at all.", 3, "25-Jun-2021", 8),
    ("1007", "Omeprazole", "Stomach Ulcer", "Improved my quality of life.", 9, "30-Jul-2021", 150),
    ("1008", "Losartan", "Insomnia", "Experienced severe side effects.", 4, "18-Aug-2021", 56),
    ("1009", "Metformin", "Diabetes", "This drug has been life-changing.", 10, "01-Sep-2021", 300),
    ("1010", "Ibuprofen", "Pain", "Good relief but with some side effects.", 7, "05-Oct-2021", 20)
]

new_data = spark.createDataFrame(new_input_data, schema=schema)

# Preprocess new data (ensure the same preprocessing as the training data)
new_data = new_data.na.fill({"uniqueID": 0, "drugName": "unknown", "condition": "unknown", "review": "unknown", "date": 0, "usefulCount": 0})
new_data = new_data.filter(col("rating").isNotNull()).filter(col("rating").cast("int").isNotNull())

# Load the saved pipeline for feature transformations
pipeline_path = "/dbfs/FileStore/pipeline/"
pipeline_model = PipelineModel.load(pipeline_path)

# Transform the new data
transformed_new_data = pipeline_model.transform(new_data)

# Load the trained logistic regression model from MLflow
logged_model_uri = "runs:/d0ecb830bcfb400c847354d6728ac4b5/model"  # Replace <run_id> with your actual run ID
lr_model = mlflow.spark.load_model(logged_model_uri)

# Make predictions on new data
predictions = lr_model.transform(transformed_new_data)

# Display predictions
simple = predictions.select( "drugName", "condition", "rating","prediction").dropDuplicates()
display(simple)

