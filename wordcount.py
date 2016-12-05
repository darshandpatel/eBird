import pyspark
from pyspark import SparkConf, SparkContext
conf = (SparkConf()
         .setMaster("local")
         .setAppName("eBird")
         .set("spark.executor.memory", "1g"))
sc = SparkContext(conf = conf)
words = sc.textFile("C:\Users\SPS\Documents\http_output_113.txt")
s1,s2 = words.randomSplit([8,2],10)
print s1.count()
print s2.cout
print words.count()
