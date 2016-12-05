import pyspark
from pyspark import SparkConf, SparkContext
conf = (SparkConf()
         .setMaster("local")
         .setAppName("eBird")
         .set("spark.executor.memory", "1g"))
sc = SparkContext(conf = conf)
seed = 17
labelled = sc.textFile("C:\Users\SPS\Documents\eBird\\training.bz2")
train, validate = labelled.randomSplit([8,2],seed)
train1, sample = validate.randomSplit([9,1],seed)
sample.saveAsTextFile("sample.csv")
