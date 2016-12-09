import random
from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel
from pyspark.mllib.regression import LabeledPoint
from pyspark import SparkConf, SparkContext
import sys
from pyspark.mllib.tree import RandomForest, RandomForestModel


class ModelTraining:

    @staticmethod
    def handle_class_imbalance(values):
        try:
            if values[0] == 'x':
                return True
            elif values[0] == '0':
                random_value = random.randint(1, 4)
                # Half the data
                if random_value == 1 or random_value == 2:
                    return True
                else:
                    return False
            elif values[0] == '?':
                if random.randint(0, 1) == 1:
                    return True
            else:
                # All positive cases
                return True
        except:
            return False

    @staticmethod
    def parse_input(values):
        values = [float(x) for x in values]
        return LabeledPoint(values[0], values[1:])

    @staticmethod
    def cal_accuracy(sc, result_rdd):

        true_pos = sc.accumulator(0)
        false_pos = sc.accumulator(0)
        true_neg = sc.accumulator(0)
        false_neg = sc.accumulator(0)
        total = sc.accumulator(0)
        result_rdd.foreach(lambda line : ModelTraining.validate(line, true_pos, true_neg, false_pos, false_neg, total))

        print 'true_pos : ',true_pos.value
        print 'true_neg : ' , true_neg.value
        print 'false_pos : ', false_pos.value
        print 'false_neg : ', false_neg.value
        print 'total : ' , total.value
        print 'accuracy : ', float(true_pos.value+true_neg.value)/total.value

    @staticmethod
    def validate(line, true_pos, true_neg, false_pos, false_neg, total):
        parts = line.split(',')
        total.add(1)
        if float(parts[1]) == 0:
            if int(parts[2]) == 0:
                true_neg.add(1)
            else:
                false_pos.add(1)
        elif float(parts[1]) == 1:
            if int(parts[2]) == 1:
                true_pos.add(1)
            else:
                false_neg.add(1)

    @staticmethod
    def train_logistic_regression(train_rdd):
        # Build Model
        model = LogisticRegressionWithLBFGS.train(train_rdd, regParam=0.005, regType='l1')
        return model

    @staticmethod
    def train_random_forest(train_rdd):
        parsed_train_rdd = train_rdd.map(ModelTraining.parse_input)
        model = RandomForest.trainClassifier(parsed_train_rdd, numClasses=2, categoricalFeaturesInfo={},
                                             numTrees=8, featureSubsetStrategy="auto",
                                             impurity='gini', maxDepth=15, maxBins=32)
        return model

if __name__ == "__main__":

    args = sys.argv
    input_path = args[1]
    conf = SparkConf()
    sc = SparkContext(conf=conf)
    prediction_rdd = sc.textFile(input_path)
    ModelTraining.cal_accuracy(sc, prediction_rdd)

    '''
    processed_train_rdd = train_rdd.filter(lambda x: DataExploration.filter_values_by_target_class(x)).map(
        lambda x: DataExploration.swap_target(x)).map(lambda x: DataExploration.custom_function(x))
    # srdd = rdd.flatMap(lambda x: DataExploration.custom_function(x))

    ModelTraining.train_logistic_regression(processed_train_rdd, "/Users/Darshan/Documents/MapReduce/FinalProject/Model")

    '''

