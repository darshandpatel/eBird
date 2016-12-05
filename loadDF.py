import pyspark
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
conf = (SparkConf()
         .setMaster("local")
         .setAppName("eBird")
         .set("spark.executor.memory", "1g"))
sc = SparkContext(conf = conf)
sqc = SQLContext(sc)
sample = sc.textFile("./sample/part-00000").persist()
sampledf = sample.map(lambda x : x.split(",")).toDF()
newsampledf = sampledf.drop(sampledf._1)
newsampledf.filter(sampledf._18 != "?").show(10)
