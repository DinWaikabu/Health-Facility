#import library
from pyspark.sql import SparkSession
from pyspark.sql import Window
from pyspark import SparkConf
from pyspark.sql.types import * 
from pyspark import SparkContext
from pyspark.sql.functions import *
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
import numpy as np
import matplotlib.pyplot as plt

#config
spark = SparkSession.builder.master("yarn").appName("Peserta_Akses").getOrCreate()
#load Data
fktpDataFrame = spark.read.csv("/user/hive/warehouse/db_ml.db/kunjungan_fktp/" , header=False, sep="|")

#columns = ['idpeserta','nofpk','nosjp','jnsklaimreg','nmproplayan','nmkrlayan','nmkclayan','nmdati2layan',
# 'kdppklayan','tglpelayanan','klsrawat','jenisppklayan','typeppklayan','nmpropperujuk','nmkrperujuk','nmkcperujuk','nmdati2perujuk',
# 'kdppkperujuk','typeppkperujuk','nmtkp','kddiagmasuk','nmdiagmasuk',
# 'kddiagprimer','nmdiagprimer','kdinacbgs','nminacbgs','politujsjp','byversjp']

columns = ["kdppkkunjungan", "nmppkkunjungan", "kdkrkunjungan", "nmkrkunjungan", "kdkckunjungan", "nmkckunjungan",
"kdpropkunjungan", "nmpropkunjungan", "kddati2kunjungan", "nmdati2kunjungan", "nmjnsppkkunjungan", "kdpoli", "nmpoli", "kdtkp", "nmtkp", "tgl_kunjungan",
"kd_diagnosa", "nm_diagnosa", "kdppkrujukan", "nmppkrujukan"]

#Rename columns
for c,n in zip(fktpDataFrame.columns,columns):
    fktpDataFrame=fktpDataFrame.withColumnRenamed(c,n)

### Data Prep
# cut datafarme >= 2020-07-01
def manipulationString(data):
    df = data.withColumn("Periode", substring("tgl_kunjungan", 1,10))
    df = df.withColumn("tgl_kunjungan2date", to_date(unix_timestamp(df["Periode"], "yyyy-MM-dd").cast("timestamp")))
    df = df.filter(df.Periode >= "2020-07-01")
    return df
#get top 20 diagnosa
def top20(data):
    df = data.groupBy("kdppkkunjungan","kd_diagnosa"
                     ).agg(count("kd_diagnosa").alias("Jumlahkunjungan"))
    window = Window.partitionBy(df['kdppkkunjungan']).orderBy(df['Jumlahkunjungan'].desc())
    return df.select('*', rank().over(window).alias('rank')).filter(col('rank') <= 20)
    
#pivot data
def pivotData(data):
    dataFinal = data.groupBy('kdppkkunjungan').pivot('kd_diagnosa').agg(sum('Jumlahkunjungan'))
    return dataFinal.fillna(0)


dataFrame = pivotData(top20(manipulationString(fktpDataFrame)))
dataFrame.show()

## Set vector assembler
variabel = dataFrame.columns
variabel.remove('kdppkkunjungan')
assembler = VectorAssembler(inputCols=[*variabel] ,outputCol="features")
df = assembler.transform(dataFrame)

#get K Optimal
cost = np.zeros(20)
for k in range(2,20):
    kmeans = KMeans().setK(k).setSeed(1).setFeaturesCol("features")
    model = kmeans.fit(df.sample(False,0.1, seed=42))
    cost[k] = model.computeCost(df) # requires Spark 2.0 or later
    
    
fig, ax = plt.subplots(1,1, figsize =(8,6))
ax.plot(range(2,20),cost[2:20])
ax.set_xlabel('k')
ax.set_ylabel('cost')

#Implementasi Model
kmeans = KMeans(
    featuresCol=assembler.getOutputCol(), 
    predictionCol="cluster", k=4)
model = kmeans.fit(df)
print ("Model is successfully trained!")

prediction = model.transform(df)#cluster given data
print('Member Cluster')
#member cluster
prediction.groupBy("cluster").count().orderBy("cluster").show()
#detail data with predcit
prediction.select('kdppkkunjungan', 'cluster').show(truncate=False)
