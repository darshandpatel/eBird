import pyspark
from pyspark import SparkConf, SparkContext
from pyspark.sql import column
def customfunction(x):
    if x == "?":
        x = "0"
    return x
from pyspark.sql import SQLContext
conf = (SparkConf()
         .setMaster("local")
         .setAppName("eBird")
         .set("spark.executor.memory", "1g"))
sc = SparkContext(conf = conf)
sqc = SQLContext(sc)
sample = sc.textFile("./sample/part-00000").persist()
sampledf = sample.map(lambda x : x.split(",")).toDF()
#to drop a column
#newsampledf = sampledf.drop(sampledf._1)

#to filter rows
#newsampledf.filter(sampledf._18 != "?").show(10)

#to replace ? with 0
sampledf.replace("?","0").show(10)
