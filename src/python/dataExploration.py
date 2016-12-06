from pyspark import SparkConf, SparkContext


class DataExploration:

    def __init__(self):
        self.conf = SparkConf().setMaster("local").setAppName("Machine Learning")  # (SparkConf().setMaster("local").setAppName("eBird"))
        self.sc = SparkContext(conf=self.conf)

    def splitInputData(self, inputPath, outputPath):
        seed = 17
        labelled = self.sc.textFile(inputPath)
        headers = labelled.first()
        print headers
        modifiedLabels = labelled.subtract(self.sc.parallelize(headers))
        train, validate = modifiedLabels.randomSplit([9, 1], seed)
        train1, sample = validate.randomSplit([9, 1], seed)
        sample.saveAsTextFile(outputPath)

if __name__ == "__main__":
    inputPath = "/Users/Darshan/Documents/MapReduce/FinalProject/labeled.csv.bz2"
    outputPath = "/Users/Darshan/Documents/MapReduce/FinalProject/LabeledSample"
    dataExploration = DataExploration()
    dataExploration.splitInputData(inputPath=inputPath, outputPath=outputPath)