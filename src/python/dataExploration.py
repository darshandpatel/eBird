from pyspark import SparkContext, SparkConf

conf = SparkConf().setMaster("local").setAppName("Machine Learning")
sc = SparkContext(conf=conf)

textFile = sc.textFile("/Users/Darshan/Documents/MapReduce/FinalProject/labeled.csv.bz2")

print textFile.first()