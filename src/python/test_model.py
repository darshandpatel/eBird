from pyspark import SparkConf, SparkContext
from sklearn import linear_model
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import sys


class DataExploration:

    def __init__(self):
        self.conf = SparkConf().setMaster("local").setAppName("eBird")
        self.sc = SparkContext(conf=self.conf)
        self.models_broadcast = None

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
        return self.sc.textFile(file_path, 2)

    @staticmethod
    def print_information(rdd):
        print rdd.take(10)

    def broadcast_models(self, models):
        self.models_broadcast = self.sc.broadcast(models)

    def predict_on_map(self, test):
        test.map(self.average_prediction)

    def read_model_files(self, file_path):
        return self.sc.pickleFile(file_path)

    @staticmethod
    def apply_linear_regression(values):
        clf = linear_model.LinearRegression()
        xs = []
        ys = []
        for value in values:
            update_value = DataExploration.type_cast(value)
            xs.append([update_value[0], update_value[1], update_value[2]])
            ys.append([update_value[3]])

        clf = clf.fit(np.asarray(xs), np.asarray(ys))
        # yield  clf
        print type(clf)
        return [clf]

    @staticmethod
    def random_forest_average_prediction(models_broadcast, lines):

        xs = []
        for line in lines:
            update_value = DataExploration.type_cast(line)
            xs.append(update_value)

        total_predication= np.zeros(len(xs))
        for model in models_broadcast.value:
            prediction = model.predict(np.asarray(xs))
            total_predication = np.add(total_predication, prediction)

        yield (total_predication/len(models_broadcast.value)).tolist()

    @staticmethod
    def linear_regression_average_prediction(models_broadcast, line):

        xs = []
        update_value = DataExploration.type_cast(line)
        xs.append(update_value)
        total_prediction = 0
        for model in models_broadcast.value:
            prediction = model.predict(np.asarray(xs))[0][0]
            print prediction
            #print "prediction : " + prediction
            total_prediction += prediction

        return total_prediction/len(models_broadcast.value)

    @staticmethod
    def apply_sample_function(values):
            for value in values:
                yield value[0]

    @staticmethod
    def apply_random_forest(values):

        clf = RandomForestClassifier(n_estimators=10)
        xs = []
        ys = []
        for value in values:
            update_value = DataExploration.type_cast(value)
            xs.append([update_value[0], update_value[1], update_value[2]])
            ys.append([update_value[3]])

        clf = clf.fit(np.asarray(xs), np.asarray(ys))
        print type(clf)
        return [clf]

    @staticmethod
    def type_cast(value):

        parts = str(value).split(',')
        parts = map(lambda x : float(x), parts)
        return parts


if __name__ == "__main__":
    args = sys.argv
    print args
    print 'Hello'

    if args[1] == "m":
        input_path = args[2]
        output_path = args[3]
        print 'input_path : ',input_path
        print 'output_path : ', output_path
        dataExploration = DataExploration()
        train = dataExploration.read_sample_training(input_path)
        #DataExploration.print_information(train)
        models = train.mapPartitions(DataExploration.apply_linear_regression)
        print models.count()
        models.saveAsPickleFile("/Users/Darshan/Documents/MapReduce/FinalProject/models")
        #modelsList = [model for model in models.toLocalIterator()]
        print models.collect()
    else:
        test_path = args[2]
        model_path = args[3]
        output_path = args[4]
        #Read Pickle file
        dataExploration = DataExploration()
        models = dataExploration.read_model_files(model_path)
        print models.collect()
        modelsList = [model for model in models.toLocalIterator()]
        dataExploration.broadcast_models(modelsList)

        sc = dataExploration.sc
        models_broadcast = dataExploration.models_broadcast
        test = dataExploration.read_sample_training(test_path)
        # prediction = test.mapPartitions(lambda lines: average_prediction(models_broadcast, lines))
        prediction = test.map(lambda lines: DataExploration.linear_regression_average_prediction(models_broadcast, lines))
        print prediction.collect()
        prediction.saveAsTextFile(output_path)