from pyspark import SparkConf, SparkContext


class DataExploration:

    def __init__(self):
        self.conf = SparkConf().setMaster("local").setAppName("eBird")
        self.sc = SparkContext(conf=self.conf)

    def split_input_data(self, input_path, output_path):
        seed = 17
        labelled = self.sc.textFile(input_path)
        headers = labelled.first()
        print headers
        modified_labels = labelled.subtract(self.sc.parallelize(headers))
        train, validate = modified_labels.randomSplit([9, 1], seed)
        train1, sample = validate.randomSplit([9, 1], seed)
        sample.saveAsTextFile(output_path)

    def read_sample_training(self, file_path):
        return self.sc.textFile(file_path)

    def print_information(self, rdd):
        print rdd.take(10)

if __name__ == "__main__":

    '''
    input_path = "/Users/Darshan/Documents/MapReduce/FinalProject/labeled.csv.bz2"
    output_path = "/Users/Darshan/Documents/MapReduce/FinalProject/LabeledSample"
    dataExploration = DataExploration()
    dataExploration.split_input_data(input_path=input_path, output_path=output_path)
    '''
    dataExploration = DataExploration()
    rdd = dataExploration.read_sample_training("/Users/Darshan/Documents/MapReduce/FinalProject/LabeledSample/")
    dataExploration.print_information(rdd)

