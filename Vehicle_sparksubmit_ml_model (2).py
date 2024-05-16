#!/usr/bin/env python
# coding: utf-8

# In[1]:


import findspark
findspark.init()
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import *

spark = SparkSession.builder.appName("UsedCarPricePrediction").getOrCreate()
sc = spark.sparkContext

# Reading csv file in pyspark dataframe:
df = spark.read.csv(r"X:\spark1\vehicles.csv",header=True, inferSchema=True)

df.printSchema()
df.count()

#Drop the independent columns :
columns_to_delete = ['id', 'url', 'region_url','VIN','image_url','description','county','lat','long','posting_date','size','state']
df1 = df.drop(*columns_to_delete)

# Show column after deleting:
df1.columns

#drop duplicated records :
df2 = df1.distinct()
df2.count()

#Calculate the percentage of null values for each column :
null_counts = df2.select([sum(col(column).isNull().cast('int')).alias(column) for column in df2.columns])
total_rows = df2.count()
null_percentages = null_counts.select([((col(column) / total_rows) * 100).alias(column + "_null_percentage") for column in df2.columns])
null_percentages.show() # show null value %

#Drop rows with null values in specified columns :
df3 = df2.na.drop(subset=['region','price','year','model','odometer','manufacturer','transmission','title_status','fuel'])
df3.count()

#Handling Missing Values with Categorical Encoding :
df4 = df3.fillna('unknown')

#handling cloumns :

#manufacturer :
# Define the list of top 20 manufacturers
manufacturer_values = ['nissan','honda','chevrolet','mercedes-benz','ram','dodge','ford','jeep','toyota','bmw','subaru','volkswagen','kia','cadillac','hyundai','lexus','audi','chrysler','acura','buick']

# Use when function to update the 'manufacturer' column
df5 = df4.withColumn('manufacturer', 
                   when(df4['manufacturer'].isin(manufacturer_values), df4['manufacturer'])
                   .otherwise('others'))

# region :
# Count the occurrences of each region value
manufacturer_counts = df5.groupBy('region').count()

# Sort the counts in descending order and select the top 50 region
top_manufacturers = manufacturer_counts.orderBy('count', ascending=False).limit(50)

# Extract the top 50 region values
manufacturer_values = [row['region'] for row in top_manufacturers.collect()]

# Use when function to update the 'region' column
df6 = df5.withColumn('region', 
                     when(df5['region'].isin(manufacturer_values), df5['region'])
                     .otherwise('others'))

#model :
# Count the occurrences of each model value
manufacturer_counts = df6.groupBy('model').count()

# Sort the counts in descending order and select the top 50 model
top_manufacturers = manufacturer_counts.orderBy('count', ascending=False).limit(50)

# Extract the top 50 model values
manufacturer_values = [row['model'] for row in top_manufacturers.collect()]

# Use when function to update the 'model' column
df7 = df6.withColumn('model',
                     when(df6['model'].isin(manufacturer_values), df6['model'])
                     .otherwise('others'))


# transmission :
names_to_match = ['automatic','manual','other','unknown'] 
df8 = df7.filter((col("transmission").isin(names_to_match)) )

#year :
#converting year, odometer, price column type to integer type:

df9 = df8.withColumn("year", col('year').cast("int"))
df10 = df9.withColumn("odometer", col('odometer').cast("int"))
df11 = df10.withColumn("price", col('price').cast("int"))
#df11.printSchema()

# handling outliers :
#price:
# Calculate quartiles

price_percentiles = df11.approxQuantile("price", [0.15, 0.75], 0.01)
price_percentile15 = price_percentiles[0]
price_percentile75 = price_percentiles[1]

# Calculate IQR and upper/lower limits

price_iqr = price_percentile75 - price_percentile15
price_upper_limit = price_percentile75 + 1.5 * price_iqr
price_lower_limit = price_percentile15

# Filter DataFrame based on limits
df12 = df11.filter((col("price") < price_upper_limit) & (col("price") > price_lower_limit))


#odometer:
# Calculate percentiles
odometer_percentiles = df12.approxQuantile("odometer", [0.05, 0.25, 0.75], 0.01)
odometer_percentile05 = odometer_percentiles[0]
odometer_percentile25 = odometer_percentiles[1]
odometer_percentile75 = odometer_percentiles[2]

# Calculate IQR and upper/lower limits
odometer_iqr = odometer_percentile75 - odometer_percentile25
odometer_upper_limit = odometer_percentile75 + 1.5 * odometer_iqr
odometer_lower_limit = odometer_percentile05

# Filter DataFrame based on limits
df13 = df12.filter((col("odometer") < odometer_upper_limit) & (col("odometer") > odometer_lower_limit))

#year : removing year before 1996 based on barplot distribution :
# Filter DataFrame based on the condition
df14 = df13.where(df13['year'] > 1996)

# Drop records where year column has a value of 2022
df15 = df14.filter(df['year'] != 2022)

# adding new column 'car_age' based on purchase year and till 2022
df16 = df15.withColumn('car_age', 2024 - col('year'))

#droping year :
df17 = df16.drop('year')
df17.printSchema()


# In[2]:


from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler,StandardScaler
from pyspark.ml.regression import LinearRegression,RandomForestRegressor,DecisionTreeRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator


# In[34]:


train, test = df17.randomSplit([0.8, 0.2],seed=23)
numerical=["odometer","car_age"]
numerical_vector_assembler = VectorAssembler(inputCols=numerical,outputCol='numerical_feature_vector')

train = numerical_vector_assembler.transform(train)
test = numerical_vector_assembler.transform(test)

scaler = StandardScaler(inputCol='numerical_feature_vector',outputCol='scaled_numerical_feature_vector',withStd=True, withMean=True)

fit_scaler = scaler.fit(train)

train = fit_scaler.transform(train)
test = fit_scaler.transform(test)

indexer = StringIndexer(inputCols=['manufacturer','model','condition','cylinders','fuel','title_status','transmission','drive','type','paint_color','region'],
                        outputCols=['manufacturer_index','m_i','co_i','cy_i','f_i','ts_i','tr_i','d_i','ty_i','p_i','r_i'],handleInvalid="keep")

fit_indexer = indexer.fit(train)
train = fit_indexer.transform(train)
test = fit_indexer.transform(test)


one_hot_encoder = OneHotEncoder(inputCols=['manufacturer_index','m_i','co_i','cy_i','f_i','ts_i','tr_i','d_i','ty_i','p_i','r_i'],
                                outputCols=['manufacturer_index_h','m_i_h','co_i_h','cy_i_h','f_i_h','ts_i_h','tr_i_h','d_i_h','ty_i_h','p_i_h','r_i_h'])

fit_one_hot_encoder = one_hot_encoder.fit(train)

train = fit_one_hot_encoder.transform(train)
test = fit_one_hot_encoder.transform(test)

assembler = VectorAssembler(inputCols=['scaled_numerical_feature_vector',
                                       'manufacturer_index_h','m_i_h','co_i_h','cy_i_h','f_i_h','ts_i_h','tr_i_h','d_i_h','ty_i_h','p_i_h','r_i_h'],
                            outputCol='final_feature_vector')

train = assembler.transform(train)
test = assembler.transform(test)


# In[35]:


lr = LinearRegression(featuresCol='final_feature_vector',labelCol='price')
lr=lr.fit(train)

# Step 7: Make predictions on the testing data
predictions = lr.transform(test)

# Step 8: Evaluate the model
evaluator = RegressionEvaluator(labelCol="price", predictionCol="prediction", metricName="r2")
r2_li = evaluator.evaluate(predictions)
mse_li = evaluator.evaluate(predictions, {evaluator.metricName: "mse"})
mae_li = evaluator.evaluate(predictions, {evaluator.metricName: "mae"})
rmse_li = evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})

# Print the evaluation metrics
print("R2 on test data = %g" % r2_li)
print("Mean squared error:", mse_li)
print("Mean absolute error:", mae_li)
print("Root mean squared error:", rmse_li)

# Show predictions
predictions.select("final_feature_vector", "price", "prediction").show()


# In[5]:


rf = RandomForestRegressor(featuresCol='final_feature_vector',labelCol='price')
rf = rf.fit(train)

predictions = rf.transform(test)

# Evaluate the model
evaluator = RegressionEvaluator(labelCol="price", predictionCol="prediction", metricName="r2")

# Evaluate the model
r2_rf= evaluator.evaluate(predictions)
mse_rf= evaluator.evaluate(predictions, {evaluator.metricName: "mse"})
mae_rf= evaluator.evaluate(predictions, {evaluator.metricName: "mae"})
rmse_rf= evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})

# Print the evaluation metrics
print("R2 on test data = %g" % r2_rf)
print("Mean squared error:", mse_rf)
print("Mean absolute error:", mae_rf)
print("Root mean squared error:", rmse_rf)

# Show predictions
predictions.select("final_feature_vector", "price", "prediction").show()


# In[5]:


# Define Decision Tree Regressor
dt = DecisionTreeRegressor(featuresCol='final_feature_vector',
                      labelCol='price')

# Train the model
dt = dt.fit(train)

# Make predictions
predictions = dt.transform(test)

# Evaluate the model
evaluator = RegressionEvaluator(labelCol="price", predictionCol="prediction", metricName="r2")
r2_dt= evaluator.evaluate(predictions)
mse_dt= evaluator.evaluate(predictions, {evaluator.metricName: "mse"})
mae_dt= evaluator.evaluate(predictions, {evaluator.metricName: "mae"})
rmse_dt= evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})

# Print the evaluation metrics
print("R2 on test data = %g" % r2_dt)
print("Mean squared error:", mse_dt)
print("Mean absolute error:", mae_dt)
print("Root mean squared error:", rmse_dt)

# Show predictions
predictions.select("final_feature_vector", "price", "prediction").show()


# In[6]:


# Define Ridge Regression model
ridge = LinearRegression(featuresCol="final_feature_vector", labelCol="price", elasticNetParam=0.0, regParam=0.5) # regParam is the regularization parameter for Ridge Regression

# Train the model
ridge_model = ridge.fit(train)

# Make predictions
predictions = ridge_model.transform(test)

# Evaluate the model
evaluator = RegressionEvaluator(labelCol="price", predictionCol="prediction", metricName="r2")
r2_ri= evaluator.evaluate(predictions)
mse_ri= evaluator.evaluate(predictions, {evaluator.metricName: "mse"})
mae_ri= evaluator.evaluate(predictions, {evaluator.metricName: "mae"})
rmse_ri= evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})
print("R2 on test data = %g" % r2_ri)
print("Mean squared error:", mse_ri)
print("Mean absolute error:", mae_ri)
print("Root mean squared error:", rmse_ri)
# Show predictions
predictions.select("final_feature_vector", "price", "prediction").show()


# In[7]:


# Define Ridge Regression model
lasso = LinearRegression(featuresCol="final_feature_vector", labelCol="price", elasticNetParam=1.0, regParam=0.1) # regParam is the regularization parameter for Ridge Regression

# Train the model
las_model = lasso.fit(train)

# Make predictions
predictions = las_model.transform(test)

# Evaluate the model
evaluator = RegressionEvaluator(labelCol="price", predictionCol="prediction", metricName="r2")
r2_las= evaluator.evaluate(predictions)
mse_las= evaluator.evaluate(predictions, {evaluator.metricName: "mse"})
mae_las= evaluator.evaluate(predictions, {evaluator.metricName: "mae"})
rmse_las= evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})
print("R2 on test data = %g" % r2_las)
print("Mean squared error:", mse_las)
print("Mean absolute error:", mae_las)
print("Root mean squared error:", rmse_las)
# Show predictions
predictions.select("final_feature_vector", "price", "prediction").show()


# In[7]:


# Train GBoost model
gbt = GBTRegressor(featuresCol="final_feature_vector", labelCol="price")
gbt_model = gbt.fit(train)

# Save the trained GBT model
#model_path = "X:/FINAL/GBT_model2020"
#gbt_model.save(model_path)

# Make predictions
predictions = gbt_model.transform(test)

# Evaluate the model
evaluator = RegressionEvaluator(labelCol="price", predictionCol="prediction", metricName="r2")
r2_gb= evaluator.evaluate(predictions)
mse_gb= evaluator.evaluate(predictions, {evaluator.metricName: "mse"})
mae_gb= evaluator.evaluate(predictions, {evaluator.metricName: "mae"})
rmse_gb= evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})
print("R2 on test data = %g" % r2_gb)
print("Mean squared error:", mse_gb)
print("Mean absolute error:", mae_gb)
print("Root mean squared error:", rmse_gb)
# Show predictions
predictions.select("final_feature_vector", "price", "prediction").show()


# In[ ]:


from pyspark.sql.types import StructType, StructField, StringType, DoubleType

# Define the schema for the DataFrame
schema = StructType([
    StructField("Model", StringType(), True),
    StructField("R2", DoubleType(), True),
    StructField("Mse", DoubleType(), True),
    StructField("Mae",DoubleType(),True),
    StructField("Rmse",DoubleType(),True)
])

# Define the data for the DataFrame
data = [
    ("Linear",r2_li,mse_li,mae_li,rmse_li),
    ("RandomForest",r2_rf,mse_rf,mae_rf,rmse_rf),
    ("DecisionTree",r2_dt,mse_dt,mae_dt,rmse_dt),
    ("Ridge",r2_ri,mse_ri,mae_ri,rmse_ri),
    ("Lasso",r2_las,mse_las,mae_las,rmse_las),
    ("GradientBoosting",r2_gb,mse_gb,mae_gb,rmse_gb)
]

# Create a DataFrame using the schema and data
df_model = spark.createDataFrame(data, schema)

# Show the DataFrame
#df_model.show()

from pyspark.sql.functions import round

# Round off the values in the DataFrame to four decimal places
df_rounded = df_model.select(
    df_model["Model"],
    round(df_model["R2"],2).alias("R2"),
    round(df_model["Mse"],4).alias("Mse"),
    round(df_model["Mae"],4).alias("Mae"),
    round(df_model["Rmse"],4).alias("Rmse")
)

# Show the rounded DataFrame
df_rounded.show()


# ### Load the saved model :

# In[40]:


#model_path = "X:/FINAL/GBT_model"
#gbt_model1=gbt_model.load(model_path)


# In[31]:


import pyspark.sql.functions as F
from pyspark.sql import SparkSession
import gradio as gr
from pyspark.ml import PipelineModel


# ## User input interface :

# In[67]:


# Define a function to predict car prices:

def predict_price(car_age, odometer,manufacturer,model,condition,cylinders,fuel,title_status,transmission,drive,ttype,paint_color,region):
    # Create a DataFrame with the user input
    input_df = spark.createDataFrame([(car_age, odometer,manufacturer,model,condition,cylinders,fuel,title_status,transmission,drive,ttype,paint_color,region)], ['car_age','odometer','manufacturer','model','condition','cylinders','fuel','title_status','transmission','drive','type','paint_color','region'])
    
    
   # numerica=["odometer","car_age"]

    input_numerical = numerical_vector_assembler.transform(input_df)


    scaled_df = fit_scaler.transform(input_numerical)
    #scaled_df.printSchema()

    
    indexed_df = fit_indexer.transform(scaled_df)
    #indexed_df.printSchema()

    
    encoded_df = fit_one_hot_encoder.transform(indexed_df)
    #encoded_df.printSchema()
    
    # Transform the data
    final_df = assembler.transform(encoded_df)#
    
    # Evaluate the model
   # evaluator = RegressionEvaluator(labelCol="price", predictionCol="prediction", metricName="r2")
    
    # Make predictions
    predictions_ = gbt_model.transform(final_df)
    
    # Extract predicted price
    predicted_price = predictions_.select(F.col("prediction")).first()[0]
    
    
    return predicted_price


# In[68]:


predict_price(12, 15000, "kia", "others", "good", "6 cylinders", "gas", "clean", "automatic", "4wd", "sedan", "white", "others")


# In[54]:


# Create a Gradio interface
import gradio as gr
iface = gr.Interface(fn=predict_price,
    inputs=[
    gr.Number(label="ODOMETER"),
    gr.Number(label="CAR's AGE"),
    gr.Textbox(label="MANUFACTURER"),
    gr.Textbox(label="MODEL"),
    gr.Dropdown(label="CONDITION", choices=["salvage","new", "used", "like new", "good","fair","excellent","unknown"]),
    gr.Textbox(label="CYLINDERS"),
    gr.Dropdown(label="FUEL", choices=["gas", "diesel", "hybrid", "electric", "other"]),
    gr.Dropdown(label="TITLE STATUS", choices=["lien", "missing","clean", "rebuilt", "salvage", "parts only"]),
    gr.Dropdown(label="TRANSMISSION", choices=["automatic", "manual", "other"]),
    gr.Dropdown(label="DRIVE", choices=["fwd", "rwd", "4wd","unknown"]),
    gr.Dropdown(label="TYPE", choices=["sedan","van", "mini-van","offroad","wagon","coupe","bus" , "SUV","pickup", "truck", "hatchback", "convertible", "Other"]),
    gr.Dropdown(label="PAINT COLOUR", choices=["white","orange", "grey", "green", "yellow", "silver", "purple","red", "black","brown", "blue","custom","unknown"]),
    gr.Textbox(label="REGION")
    ],
    outputs=gr.Textbox(label="Predicted Price"),
    title="Car Price Prediction",
    description="Enter the features of the car to predict its price.",
)

# Launch the interface
iface.launch(share=True)


# In[ ]:




