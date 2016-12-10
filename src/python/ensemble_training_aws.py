from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
import math
import csv
from pyspark.mllib.linalg import SparseVector
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.regression import LabeledPoint
import sys
from sklearn import linear_model, ensemble
from sklearn.neural_network import MLPClassifier

class DataExploration:

    header_dict = {}
    drop_list = ["SAMPLING_EVENT_ID", "LOC_ID", "YEAR", "DAY", "COUNTRY", "STATE_PROVINCE", "COUNTY", "COUNT_TYPE", "OBSERVER_ID",
                 "ELEV_GT","ELEV_NED","GROUP_ID","BAILEY_ECOREGION", "OMERNIK_L3_ECOREGION","SUBNATIONAL2_CODE", "LATITUDE", "LONGITUDE"]
    drop_multiples_list = ["NLCD", "CAUS_PREC0", "CAUS_PREC1", "CAUS_SNOW0", "CAUS_SNOW1", "CAUS_TEMP_AVG0", "CAUS_TEMP_AVG1"
                           , "CAUS_TEMP_MIN0", "CAUS_TEMP_MIN1", "CAUS_TEMP_MAX0", "CAUS_TEMP_MAX1"]
    protocol_list = ["P20", "P21", "P22", "P23", "P34", "P35", "P39", "P40", "P41", "P44", "P45", "P46", "P47", "P48",
                     "P49", "P50", "P51", "P52", "P55", "P56"]
    birds_column_ids = None
    drop_column_ids = []
    target_ID = 0
    mean = []
    variance = []

    def __init__(self):
        self.conf = SparkConf()
        self.sc = SparkContext(conf=self.conf)
        self.sqc = SQLContext(self.sc)
        self.mrdd = None

    @staticmethod
    def get_number(n):
        try:
            s = float(n)
            return s
        except ValueError:
            #print "Value error in get number:"+n
            return 0

    @staticmethod
    def create_header_dict(h):
        header_dict = {}
        target_col_name = "Agelaius_phoeniceus"
        first_col_name = h[0]
        i = 0
        for header in h:
            if header in header_dict:
                try:
                    l = len(header_dict[header])
                    header_dict[header].append(i)
                except (AttributeError, TypeError) as e:
                    header_dict[header] = [header_dict[header], i]
            else:
                header_dict[header] = i
            i += 1
        target_id = header_dict[target_col_name]
        first_id = header_dict[first_col_name]
        DataExploration.target_ID = target_id
        try:
            l = len(first_id)
            first_id.append(target_id)
            swap_id = first_id[0]
            del first_id[0]
            header_dict[first_col_name] = first_id
            header_dict[target_col_name] = swap_id
        except TypeError:
            header_dict[target_col_name] = first_id
            header_dict[first_col_name] = target_id
        return header_dict

    @staticmethod
    def get_col_id(s):
        try:
            i = DataExploration.header_dict[s]
            return i
        except KeyError:
            print "Key Error in get col id"
            return -1

    @staticmethod
    def add_protocol_list(x):
        protocol = x[DataExploration.get_col_id("COUNT_TYPE")]
        if protocol == -1:
            return x
        for val in DataExploration.protocol_list:
            if protocol == val:
                x.append(1.0)
            else:
                x.append(0.0)
        return x

    @staticmethod
    def add_time_slot(x):
        time = DataExploration.get_number(x[DataExploration.get_col_id("TIME")])
        if time == -1:
            return x
        if 3 <= time <= 8:
            x.append(1.0)
        else:
            x.append(0.0)
        if 8 <= time <= 15:
            x.append(1.0)
        else:
            x.append(0.0)
        if 15 <= time <= 20:
            x.append(1.0)
        else:
            x.append(0.0)
        if time >= 20 or time <= 3:
            x.append(1.0)
        else:
            x.append(0.0)
        return x

    @staticmethod
    def add_elev_avg(x):
        elev_gt_colID = DataExploration.get_col_id("ELEV_GT")
        elev_ned_colID = DataExploration.get_col_id("ELEV_NED")
        if elev_gt_colID == -1 or elev_ned_colID == -1:
            return x
        avg = (DataExploration.get_number(x[elev_gt_colID]) +
               DataExploration.get_number(x[elev_ned_colID]))*1.0/2
        x.append(avg)
        return x

    @staticmethod
    def add_xyz(lx):
        long_colID = DataExploration.get_col_id("LONGITUDE")
        latt_colID = DataExploration.get_col_id("LATITUDE")
        lon = DataExploration.get_number(lx[long_colID])
        lat = DataExploration.get_number(lx[latt_colID])
        R = 6371
        x = R * math.cos(lat) * math.cos(lon)
        y = R * math.cos(lat) * math.sin(lon)
        z = R * math.sin(lat)
        lx.append(x)
        lx.append(y)
        lx.append(z)
        return lx

    @staticmethod
    def add_columns(x):
        px = DataExploration.add_protocol_list(x)
        ptx = DataExploration.add_time_slot(px)
        ptex = DataExploration.add_elev_avg(ptx)
        ptelx = DataExploration.add_xyz(ptex)
        return ptelx

    @staticmethod
    def replace_caus(x):
        prec_cid = DataExploration.header_dict["CAUS_PREC"]
        snow_cid = DataExploration.header_dict["CAUS_SNOW"]
        tavg_cid = DataExploration.header_dict["CAUS_TEMP_AVG"]
        tmin_cid = DataExploration.header_dict["CAUS_TEMP_MIN"]
        tmax_cid = DataExploration.header_dict["CAUS_TEMP_MAX"]
        month = int(max(1.0,DataExploration.get_number(DataExploration.header_dict["MONTH"])))

        if len(str(month)) == 1:
            mm = "0"+str(month)
        else:
            mm = str(month)

        precmm_cid = DataExploration.header_dict["CAUS_PREC"+mm]
        try:
            snowmm_cid = DataExploration.header_dict["CAUS_SNOW"+mm]
            snowmm = x[snowmm_cid]
        except KeyError:
            snowmm = 0
        tavgmm_cid = DataExploration.header_dict["CAUS_TEMP_AVG"+mm]
        tminmm_cid = DataExploration.header_dict["CAUS_TEMP_MIN"+mm]
        tmaxmm_cid = DataExploration.header_dict["CAUS_TEMP_MAX"+mm]

        if x[prec_cid] == "?":
            x[prec_cid] = x[precmm_cid]
        if x[snow_cid] == "?":
            x[snow_cid] = snowmm
        if x[tavg_cid] == "?":
            x[tavg_cid] = x[tavgmm_cid]
        if x[tmin_cid] == "?":
            x[tmin_cid] = x[tminmm_cid]
        if x[tmax_cid] == "?":
            x[tmax_cid] = x[tmaxmm_cid]
        cx = x
        return cx

    @staticmethod
    def replace_columns(ls):
        cx = DataExploration.replace_caus(ls)
        return cx

    @staticmethod
    def drop_columns(ls):
        col_list = []
        for col in DataExploration.drop_list:
            try:
                l = len(col)
                col_list.extend(DataExploration.header_dict[col])
            except TypeError:
                col_list.append(DataExploration.header_dict[col])

        for col in DataExploration.drop_multiples_list:
            for key in DataExploration.header_dict:
                if key.startswith(col):
                    col_list.append(DataExploration.header_dict[key])

        col_list.sort(reverse=True)
        for cid in col_list:
            new_cid = cid
            del ls[new_cid]
        return ls

    @staticmethod
    def filter_value_by_checklist_header(value):

        lx = value.split(",")
        if lx[DataExploration.get_col_id("PRIMARY_CHECKLIST_FLAG")] == "1" or \
                (lx[DataExploration.get_col_id("Agelaius_phoeniceus")] != '0' and
                         lx[DataExploration.get_col_id("Agelaius_phoeniceus")] != '?'):
            return True
        else:
            return False

    @staticmethod
    def filter_header(value):
        lx = value.split(",")
        if lx[DataExploration.get_col_id("PRIMARY_CHECKLIST_FLAG")] != "PRIMARY_CHECKLIST_FLAG":
            return True
        else:
            return False

    @staticmethod
    def swap_target(line):
        ls = line.split(",")
        tmp = ls[0]
        ls[0] = ls[DataExploration.target_ID]
        ls[DataExploration.target_ID] = tmp
        return ls

    @staticmethod
    def make_sparse_vector(drca_ls):
        index_array = []
        value_array = []
        index = 0
        for val in drca_ls:
            if index == 0:
                index += 1
                continue
            if val != 0:
                index_array.append(index)
                value_array.append(drca_ls[index])
            index+=1
        sparse_vector = LabeledPoint(drca_ls[0], SparseVector(index, index_array, value_array))
        return sparse_vector

    @staticmethod
    def catagories(x):
        list = []
        pair = ((DataExploration.get_col_id("BAILEY_ECOREGION"), x[DataExploration.get_col_id("BAILEY_ECOREGION")]), 1)
        list.append(pair)
        pair = ((DataExploration.get_col_id("OMERNIK_L3_ECOREGION"), x[DataExploration.get_col_id("OMERNIK_L3_ECOREGION")]), 1)
        list.append(pair)
        pair = ((DataExploration.get_col_id("BCR"), x[DataExploration.get_col_id("BCR")]), 1)
        list.append(pair)
        pair = ((DataExploration.get_col_id("YEAR"), x[DataExploration.get_col_id("YEAR")]), 1)
        list.append(pair)
        return list

    @staticmethod
    def find_catagories(rdd):
        print rdd.map(lambda x: x.split(",")).flatMap(lambda x: DataExploration.catagories(x)).groupByKey().keys().collect()


    @staticmethod
    def custom_function(ls, is_test):
        a_ls = DataExploration.add_columns(ls)
        rca_ls = DataExploration.replace_columns(a_ls)
        n_ls = HandleMissing.convert_into_numeric_value(rca_ls,
                                                        dict= DataExploration.header_dict,
                                                        birds_index= DataExploration.birds_column_ids,
                                                        drop_index= DataExploration.drop_column_ids)
        tn_ls = None
        if(not is_test):
            tn_ls = HandleMissing.convert_target_column_into_numeric(n_ls,
                                                                    target_index=DataExploration.get_col_id(
                                                                        "Agelaius_phoeniceus"), )
        else:
            tn_ls = n_ls
        drca_ls = DataExploration.drop_columns(tn_ls)
        sparse_vector = DataExploration.make_sparse_vector(drca_ls)
        return sparse_vector

    @staticmethod
    def set_mean(m):
        DataExploration.mean = m

    @staticmethod
    def set_variance(v):
        DataExploration.variance = v

    @staticmethod
    def normalize(record, variance):
        for i in range(0,len(record)-28):
            if variance[i] != 0.0:
                record[i] = (record[i] - DataExploration.mean[i])/math.sqrt(DataExploration.variance[i])
        return record

    @staticmethod
    def create_header(headers):
        DataExploration.header_dict = DataExploration.create_header_dict(headers)

    def split_input_data(self, input_path, output_path):
        seed = 17
        labelled = self.sc.textFile(input_path)
        headers = labelled.first()
        DataExploration.header_dict = DataExploration.create_header_dict(headers.split(","))
        print headers.split(",")
        modified_labels = labelled.subtract(self.sc.parallelize(headers))
        train, validate = modified_labels.randomSplit([9, 1], seed)
        train1, sample = validate.randomSplit([9, 1], seed)
        sample.saveAsTextFile(output_path)

    def read_sample_training(self, file_path):
        return self.sc.textFile(file_path)

    @staticmethod
    def calculate_corr(irdd):
        df = irdd.toDF()
        target_bird_id = DataExploration.get_col_id("Agelaius_phoeniceus")
        remaining_bird_ids = np.delete(DataExploration.birds_column_ids, target_bird_id)
        print "All columns"
        print df.columns
        correlation_dict = dict()
        for other_bird_id in remaining_bird_ids:
            correlation = df.stat.corr('_'+str(target_bird_id), '_'+str(other_bird_id))
            correlation_dict[(target_bird_id, other_bird_id)] = correlation
            #print target_bird_id,':',other_bird_id,':',correlation
        with open('/Users/Darshan/Documents/MapReduce/FinalProject/correlation.csv', 'w') as csv_file:
            writer = csv.writer(csv_file)
            for key, value in correlation_dict.items():
                writer.writerow([key, value])
        #cid1 = 26
        #cid2 = 20
        #v1 = df.flatMap(lambda x: Vectors.dense(DataExploration.get_number(x[cid1])))
        #v2 = df.flatMap(lambda x: Vectors.dense(DataExploration.get_number(x[cid2])))
        #print Statistics.corr(v1,v2)

    @staticmethod
    def cal_birds_column_ids():
        bird_index = []
        for key, value in DataExploration.header_dict.items():
            parts = key.split('_')
            if(len(parts) > 1 and parts[1].islower()):
                bird_index.append(value)
        DataExploration.birds_column_ids = np.array(bird_index)

    @staticmethod
    def cal_drop_column_ids():
        drop_index = []
        for col in DataExploration.drop_list:
            try:
                l = len(col)
                drop_index.extend(DataExploration.header_dict[col])
            except TypeError:
                drop_index.append(DataExploration.header_dict[col])

        for col in DataExploration.drop_multiples_list:
            for key in DataExploration.header_dict:
                if key.startswith(col):
                    drop_index.append(DataExploration.header_dict[key])

        DataExploration.drop_column_ids = drop_index
        DataExploration.drop_column_ids.sort()

    @staticmethod
    def print_information(irdd):
        plist = irdd.take(2)
        print plist
        #irdd.toDF().show(10)
        '''
        for l in plist:
            for val in l:
                print l
            print "\n"
        '''

    @staticmethod
    def count_target_column(line, zeros, ones, missing, extra, total):

        target = line.split(',')[DataExploration.get_col_id("Agelaius_phoeniceus")]
        total.add(1)
        if target != 'x' and target != 'X' and target != '?':
            try:
                value = int(target)
                if value == 0:
                    zeros.add(1)
                else:
                    ones.add(1)
            except ValueError:
                print 'Exp ************* : ',target
                extra.add(1)
        elif target == 'x' or target == 'X':
            missing.add(1)
        else:
            print 'Else ************* : ', target
            extra.add(1)


    @staticmethod
    def replicate_data(value, nbr_of_models):
        duplication = []
        for i in range(nbr_of_models.value):
            duplication.append((i, value))
        return duplication

    def sparse_test(self, srdd):
        sprdd = srdd.filter(lambda x: DataExploration.filter_value_by_checklist_header(x))\
            .map(lambda x: DataExploration.swap_target(x)).\
            map(lambda x: DataExploration.custom_function(x, False))
        print sprdd.collect()[0]

    @staticmethod
    def train_model(train_data):
        print "Key : " + str(train_data[0])
        labels = []
        features = []
        for data in train_data[1]:
            labels.append(data.label)
            features.append(data.features.toArray())
            # sps_acc = sps_acc + sps.coo_matrix((d, (r, c)), shape=(rows, cols))
        labels = np.array(labels)
        features = np.array(features)

        if int(train_data[0]) == 0:
            return ModelTraining.train_sklean_neural_network(labels, features)
        elif int(train_data[0]) == 1:
            return ModelTraining.train_sklean_random_forest(labels, features)
        elif int(train_data[0]) == 2:
            return ModelTraining.train_sklean_gradient_trees(labels, features)
        elif int(train_data[0]) == 3:
            return ModelTraining.train_sklean_logistic_regression(labels, features)
        elif int(train_data[0]) == 4:
            return ModelTraining.train_sklean_adaboost(labels, features)
        else:
            return ModelTraining.train_sklean_logistic_regression(labels, features)
        #return [1]


    def perform_distributed_ml(self, train_rdd, model_path):

        processed_train_rdd = (train_rdd.filter(lambda x: DataExploration.filter_value_by_checklist_header(x)). \
                               map(lambda x: DataExploration.swap_target(x)). \
                               filter(lambda x: ModelTraining.handle_class_imbalance(x)). \
                               map(lambda x: DataExploration.custom_function(x, False)))

        #print "Actual Count : " + str(processed_train_rdd.count())

        nbr_of_models = self.sc.broadcast(4)
        replicated_train_rdd = processed_train_rdd.flatMap(lambda x: DataExploration.replicate_data(x, nbr_of_models))
        #print "Replicated Count : " + str(replicated_train_rdd.count())

        trained_group_by = replicated_train_rdd.groupByKey()
        models = trained_group_by.zipWithIndex().map(lambda x : (x[1], x[0])).mapValues(lambda x: DataExploration.train_model(x))
        #print "mapvalues : ", trained_group_by.zipWithIndex().mapValues(lambda x:x).collect()
        #print "models : ", models
            #.mapValues(DataExploration.train_model)
        #print "Map Values Count : " + str(models.count())
        #print "trained_group_by : " + str(trained_group_by.keys().count())
        #print "Models : ", models.collect()

        models.saveAsPickleFile(model_path)


class ModelTraining:

    @staticmethod
    def handle_class_imbalance(values):
        try:
            if values[0] == 'x':
                return True
            elif values[0] == '0':
                random_value = random.randint(1, 4)
                # Half the data
                if random_value == 1 or random_value == 2 or random_value == 3:
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
    def train_sklean_logistic_regression(labels, features):
        # Build Model
        log_reg = linear_model.LogisticRegression(C=1e5)
        log_reg.fit(X=features, y=labels)
        return log_reg

    @staticmethod
    def train_sklean_random_forest(labels, features):
        # Build Model
        random_forest = ensemble.RandomForestClassifier(n_estimators=15,
                                                        max_depth=25,
                                                        class_weight={1.0: 0.7, 0.0:0.3})
        random_forest.fit(X=features, y=labels)
        return random_forest

    @staticmethod
    def train_sklean_gradient_trees(labels, features):
        # Build Model
        random_forest = ensemble.GradientBoostingClassifier(loss= 'deviance',
                                                            n_estimators=100,
                                                            max_depth=5,
                                                            max_features='auto',
                                                            min_samples_leaf=500,
                                                            min_samples_split=1000,
                                                            learning_rate=0.1)
        random_forest.fit(X=features, y=labels)
        return random_forest

    @staticmethod
    def train_sklean_adaboost(labels, features):
        # Build Model
        random_forest = ensemble.GradientBoostingClassifier(loss='deviance',
                                                            n_estimators=100,
                                                            max_depth=5,
                                                            max_features='auto',
                                                            min_samples_leaf=500,
                                                            min_samples_split=1000,
                                                            learning_rate=0.1)
        random_forest.fit(X=features, y=labels)
        return random_forest

    @staticmethod
    def train_sklean_adaboost(labels, features):
        # Build Model
        adaboost = ensemble.AdaBoostClassifier(n_estimators=100,
                                               learning_rate=0.9)
        adaboost.fit(X=features, y=labels)
        return adaboost

    @staticmethod
    def train_sklean_neural_network(labels, features):
        # Build Model
        neural_network = MLPClassifier(hidden_layer_sizes=(200, 50, 20),
                                               solver='sgd',
                                               activation='logistic',
                                               learning_rate='adaptive')
        neural_network.fit(X=features, y=labels)
        return neural_network


import random
import numpy as np


class HandleMissing:

    rem_ids = []
    target_index = 0

    def __init__(self):
        self.rem_ids = []

    @staticmethod
    def convert_target_into_numeric_value(values):
        try:
            target_index = HandleMissing.target_index
            value = values[target_index]
            if value == 'x':
                values[target_index] = random.randint(2,10)
            elif value == '?':
                values[target_index] = 0.0
            else:
                values[target_index] = float(value)
        except:
            values[target_index] = 0.0
        return values

    @staticmethod
    def convert_target_into_binary_value(values):
        value = values[HandleMissing.target_index]
        if value == 0.0:
            values[HandleMissing.target_index] = 0.0
        else:
            values[HandleMissing.target_index] = 1.0
        return values

    @staticmethod
    def convert_birds_into_numeric_value(values, birds_index):
        for index in birds_index:
            if index != HandleMissing.target_index:
                value = values[index]
                try:
                    if value == 'x':
                        values[index] = float(random.randint(2,10))
                    elif value == '?':
                        values[index] = 0.0
                    else:
                        values[index] = float(value)
                except:
                    values[index] = 0.0
        return values

    @staticmethod
    def convert_remaining_into_numeric_value(values, birds_index, drop_index, dict):
        if len(HandleMissing.rem_ids) == 0:
            remaining_index = []
            for ids in dict.values():
                try:
                    remaining_index.extend(ids)
                except TypeError:
                    remaining_index.append(ids)

            temp_remaining_index = []
            for id in remaining_index:
                if id not in birds_index and id not in drop_index:
                    temp_remaining_index.append(id)

            HandleMissing.rem_ids = temp_remaining_index
            HandleMissing.rem_ids.sort()

        for index in HandleMissing.rem_ids:
            try:
                value = values[index]
                if value == 'x':
                    values[index] = float(random.randint(2,10))
                elif value == '?':
                    values[index] = 0.0
                else:
                    values[index] = float(value)
            except:
                values[index] = 0.0
        return values

    @staticmethod
    def convert_into_numeric_value(values, dict, birds_index, drop_index):
        bv_values = HandleMissing.convert_birds_into_numeric_value(values, birds_index)
        tv = HandleMissing.convert_remaining_into_numeric_value(bv_values, birds_index, drop_index, dict)
        return tv

    @staticmethod
    def convert_target_column_into_numeric(values, target_index):
        HandleMissing.target_index = target_index
        tf_values = HandleMissing.convert_target_into_numeric_value(values)
        tfb_values = HandleMissing.convert_target_into_binary_value(tf_values)
        return tfb_values

    @staticmethod
    def get_target_value(values, target_index):
        target_index = target_index
        try:
            value = values[target_index]
            if value == 'x':
                return 1.0
            elif value == '?':
                return float(0.0)
            else:
                if float(value) == 0.0:
                    return 0.0
                else:
                    return 1.0
        except:
            return float(random.randint(0,1))

if __name__ == "__main__":

    dataExploration = DataExploration()

    args = sys.argv
    input_path = args[1] # "/Users/Darshan/Documents/MapReduce/FinalProject/LabeledSample"
    val_path = args[2]  # "/Users/Darshan/Documents/MapReduce/FinalProject/LabeledSample"
    model_path = args[3] # "/Users/Darshan/Documents/MapReduce/FinalProject/Model"

    full_data_set = dataExploration.read_sample_training(input_path).persist()

    DataExploration.create_header(full_data_set.first().split(','))

    DataExploration.cal_birds_column_ids()
    DataExploration.cal_drop_column_ids()

    (train_rdd, val_rdd) = full_data_set.randomSplit([0.8, 0.2], 345)
    dataExploration.perform_distributed_ml(train_rdd, model_path)
