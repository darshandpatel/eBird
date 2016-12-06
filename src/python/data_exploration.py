from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext


class DataExploration:
    drop_list = [0, 1]
    protocol_list = ["P20", "P21", "P22", "P23", "P34", "P35", "P39", "P40", "P41", "P44", "P45", "P46", "P47", "P48", "P49", "P50", "P51", "P52", "P55", "P56"]
    protocol_colID = 11
    time_colID = 7
    elev_gt_colID = 958
    elev_ned_colID = 959
    year_colID = 4
    day_colID = 6

    def __init__(self):
        self.conf = SparkConf().setMaster("local").setAppName("eBird").set("spark.executor.memory", "2g")
        self.sc = SparkContext(conf=self.conf)
        self.sqc = SQLContext(self.sc)
        self.mrdd = None

    @staticmethod
    def get_number(n):
        try:
            s = float(n)
            return s
        except ValueError:
            return 0

    @staticmethod
    def add_protocol_list(x):
        protocol = x[DataExploration.protocol_colID]
        for val in DataExploration.protocol_list:
            if protocol == val:
                x.append("1")
            else:
                x.append("0")
        return x

    @staticmethod
    def add_time_slot(x):
        time = DataExploration.get_number(x[DataExploration.time_colID])
        if time >= 3 and time <= 8:
            x.append("1")
        else:
            x.append("0")
        if time >= 8 and time <= 15:
            x.append("1")
        else:
            x.append("0")
        if time >= 15 and time <= 20:
            x.append("1")
        else:
            x.append("0")
        if time >= 20 or time <= 3:
            x.append("1")
        else:
            x.append("0")
        return x

    @staticmethod
    def add_elev_avg(x):
        avg = (DataExploration.get_number(x[DataExploration.elev_gt_colID]) +
               DataExploration.get_number(x[DataExploration.elev_ned_colID]))*1.0/2
        x.append(str(avg))
        return x

    @staticmethod
    def add_columns(x):
        px = DataExploration.add_protocol_list(x)
        ptx = DataExploration.add_time_slot(px)
        ptex = DataExploration.add_elev_avg(ptx)
        return ptex

    @staticmethod
    def convert_year(x):
        if DataExploration.get_number(x[DataExploration.year_colID])%2 == 0:
            x[DataExploration.year_colID] = "0"
        else:
            x[DataExploration.year_colID] = "1"
        return x

    @staticmethod
    def convert_columns(x):
        yx = DataExploration.convert_year(x)
        return yx

    @staticmethod
    def drop_columns(x):
        for col in DataExploration.drop_list:
            del x[col]
        return x

    @staticmethod
    def custom_function(x):
        ls = x.split(",")
        print "LS size:"+str(len(ls))
        als = DataExploration.add_columns(ls)
        print "ALS size:"+str(len(als))
        acls = DataExploration.convert_columns(als)
        print "ACLS size:"+str(len(acls))
        dacls = DataExploration.drop_columns(acls)
        print "DACLS size:"+str(len(dacls))
        return dacls

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

    @staticmethod
    def print_information(irdd):
        print irdd.take(10)

if __name__ == "__main__":

    '''
    input_path = "/Users/Darshan/Documents/MapReduce/FinalProject/labeled.csv.bz2"
    output_path = "/Users/Darshan/Documents/MapReduce/FinalProject/LabeledSample"
    dataExploration = DataExploration()
    dataExploration.split_input_data(input_path=input_path, output_path=output_path)
    '''
    dataExploration = DataExploration()
    #rdd = dataExploration.read_sample_training("/Users/Darshan/Documents/MapReduce/FinalProject/LabeledSample/")
    rdd = dataExploration.read_sample_training("../sample/part-00000")
    srdd = rdd.map(lambda x : DataExploration.custom_function(x))
    DataExploration.print_information(srdd)
    #dataExploration.test_custom_map(rdd)
    #dataExploration.print_information(rdd)

