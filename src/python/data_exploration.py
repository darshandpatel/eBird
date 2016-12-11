from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
import numpy as np
import math
import csv
from handle_missing_value import HandleMissing
from model_training import ModelTraining
import sys
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import SparseVector
import scipy.sparse as sp

class DataExploration:

    #Column Name - ID mapping
    header_dict = {}
    #Columns to be dropped
    drop_list = ["SAMPLING_EVENT_ID", "LOC_ID", "YEAR", "DAY", "COUNTRY", "STATE_PROVINCE", "COUNTY", "COUNT_TYPE", "OBSERVER_ID",
                 "ELEV_GT","ELEV_NED","GROUP_ID","BAILEY_ECOREGION", "OMERNIK_L3_ECOREGION","SUBNATIONAL2_CODE", "LATITUDE", "LONGITUDE"]
    #Columns to be dropped after wild-card search
    drop_multiples_list = ["NLCD", "CAUS_PREC0", "CAUS_PREC1", "CAUS_SNOW0", "CAUS_SNOW1", "CAUS_TEMP_AVG0", "CAUS_TEMP_AVG1"
                           , "CAUS_TEMP_MIN0", "CAUS_TEMP_MIN1", "CAUS_TEMP_MAX0", "CAUS_TEMP_MAX1"]
    #list of possible protocols for column COUNT_TYPE
    protocol_list = ["P20", "P21", "P22", "P23", "P34", "P35", "P39", "P40", "P41", "P44", "P45", "P46", "P47", "P48",
                     "P49", "P50", "P51", "P52", "P55", "P56"]
    #List of column IDs of Birds
    birds_column_ids = None
    #list of column IDs that are dropped
    drop_column_ids = []
    target_ID = 0

    def __init__(self):
        self.conf = SparkConf()
        self.sc = SparkContext(conf=self.conf)
        self.sqc = SQLContext(self.sc)

    @staticmethod
    def get_number(n):
        #converts to number. given value will be String mostly
        try:
            s = float(n)
            return s
        except ValueError:
            #It may face error. So return 0 by default.
            return 0

    #computes Column Name - ID mapping
    @staticmethod
    def create_header_dict(h):
        header_dict = {}
        target_col_name = "Agelaius_phoeniceus"
        first_col_name = h[0]
        i = 0
        for header in h:
            if header in header_dict:
                try:
                    #it will throw error if its not a list.
                    l = len(header_dict[header])
                    #no error so its a list. So we append it. Ex. LOC_ID
                    header_dict[header].append(i)
                except (AttributeError, TypeError) as e:
                    #Error So, Dictionary contains a value but its not a List. So Create a List.
                    header_dict[header] = [header_dict[header], i]
            else:
                #No key in dictionary so add it.
                header_dict[header] = i
            i += 1
        #to swap target bird column with first column for giving as input to ML Algortihm
        target_id = header_dict[target_col_name]
        first_id = header_dict[first_col_name]
        DataExploration.target_ID = target_id
        #similar error handling mechanism as above to handle list and single elements.
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

    #Given a name of the column, get the ID.
    @staticmethod
    def get_col_id(s):
        try:
            i = DataExploration.header_dict[s]
            return i
        except KeyError:
            #Error handling. Shouldn't happen
            print "Key Error in get col id"
            return -1

    #One-Hot Encoding for COUNT_TYPE
    @staticmethod
    def add_protocol_list(x):
        protocol = x[DataExploration.get_col_id("COUNT_TYPE")]
        if protocol == -1:
            return x
        #Append 20 columns
        for val in DataExploration.protocol_list:
            if protocol == val:
                x.append(1.0)
            else:
                x.append(0.0)
        return x

    #Convert Time into 4 different slots.
    @staticmethod
    def add_time_slot(x):
        time = DataExploration.get_number(x[DataExploration.get_col_id("TIME")])
        #add 4 columns each for one slot.
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

    #Elevation GT,NED : Average it and merge the columns.
    @staticmethod
    def add_elev_avg(x):
        elev_gt_colID = DataExploration.get_col_id("ELEV_GT")
        elev_ned_colID = DataExploration.get_col_id("ELEV_NED")
        if elev_gt_colID == -1 or elev_ned_colID == -1:
            return x
        #There may still be missing values get_number handles it.
        avg = (DataExploration.get_number(x[elev_gt_colID]) +
               DataExploration.get_number(x[elev_ned_colID]))*1.0/2
        x.append(avg)
        return x

    #Convert WGS-84 coordinates to XYZ plane
    @staticmethod
    def add_xyz(lx):
        long_colID = DataExploration.get_col_id("LONGITUDE")
        latt_colID = DataExploration.get_col_id("LATITUDE")
        lon = DataExploration.get_number(lx[long_colID])
        lat = DataExploration.get_number(lx[latt_colID])
        #R - radius of earth in KM
        R = 6371
        #Standard formulae
        x = R * math.cos(lat) * math.cos(lon)
        y = R * math.cos(lat) * math.sin(lon)
        z = R * math.sin(lat)
        lx.append(x)
        lx.append(y)
        lx.append(z)
        return lx

    #Wrapper for adding columns by one-hot encoding
    #Configurable, we can add or drop anything we need easily
    @staticmethod
    def add_columns(x):
        px = DataExploration.add_protocol_list(x)
        ptx = DataExploration.add_time_slot(px)
        ptex = DataExploration.add_elev_avg(ptx)
        ptelx = DataExploration.add_xyz(ptex)
        return ptelx

    #Missing CAUS features are replaced from extended covariates.
    @staticmethod
    def replace_caus(x):
        prec_cid = DataExploration.header_dict["CAUS_PREC"]
        snow_cid = DataExploration.header_dict["CAUS_SNOW"]
        tavg_cid = DataExploration.header_dict["CAUS_TEMP_AVG"]
        tmin_cid = DataExploration.header_dict["CAUS_TEMP_MIN"]
        tmax_cid = DataExploration.header_dict["CAUS_TEMP_MAX"]
        month = int(max(1.0,DataExploration.get_number(DataExploration.header_dict["MONTH"])))
        #get month of the current record.
        if len(str(month)) == 1:
            mm = "0"+str(month)
        else:
            mm = str(month)
        #Get values from Extended Covariates using the month.
        precmm_cid = DataExploration.header_dict["CAUS_PREC"+mm]
        try:
            snowmm_cid = DataExploration.header_dict["CAUS_SNOW"+mm]
            snowmm = x[snowmm_cid]
        except KeyError:
            snowmm = 0
        tavgmm_cid = DataExploration.header_dict["CAUS_TEMP_AVG"+mm]
        tminmm_cid = DataExploration.header_dict["CAUS_TEMP_MIN"+mm]
        tmaxmm_cid = DataExploration.header_dict["CAUS_TEMP_MAX"+mm]
        #replace if missing
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

    #wrapper function for replacing/manipulating columns
    @staticmethod
    def replace_columns(ls):
        cx = DataExploration.replace_caus(ls)
        return cx

    #Drop columns present in Drop_list and multiple_Drop_list.
    @staticmethod
    def drop_columns(ls):
        #populate the IDS in column List.
        col_list = []
        #Drop_list columns are dropped.
        for col in DataExploration.drop_list:
            try:
                l = len(col)
                col_list.extend(DataExploration.header_dict[col])
            except TypeError:
                col_list.append(DataExploration.header_dict[col])
        #Wild card search performed and matching columns are dropped.
        for col in DataExploration.drop_multiples_list:
            for key in DataExploration.header_dict:
                #wild card search
                if key.startswith(col):
                    col_list.append(DataExploration.header_dict[key])
        #we need to delete from the last to avoid ID overlapping problem.
        col_list.sort(reverse=True)
        for cid in col_list:
            new_cid = cid
            del ls[new_cid]
        return ls

    @staticmethod
    def filter_value_by_checklist_header(value):
        #filter columns where PRIMIARY_CHECKLIST_FLAG is set to false
        #Keep it if the target value is 1 or x.
        lx = value.split(",")
        if lx[DataExploration.get_col_id("PRIMARY_CHECKLIST_FLAG")] == "1" or \
                (lx[DataExploration.get_col_id("Agelaius_phoeniceus")] != '0' and
                         lx[DataExploration.get_col_id("Agelaius_phoeniceus")] != '?'):
            return True
        else:
            return False

    #swap target column ID with 0, so we can easily mention the Label ID while giving it to ML algorithm
    @staticmethod
    def swap_target(line):
        ls = line.split(",")
        tmp = ls[0]
        ls[0] = ls[DataExploration.target_ID]
        ls[DataExploration.target_ID] = tmp
        return ls

    #for sparse matrix conversion.
    @staticmethod
    def make_sparse_vector(drca_ls):
        values = []
        #sparse matrix representation. (rowID, list(colID, non-zeroValue))
        index = 0
        for val in drca_ls:
            if index == 0:
                index += 1
                continue
            if val != 0:
                values.append((index, drca_ls[index]))
            index+=1
        return (drca_ls[0], index, values)

    #for sparse Vector conversion.
    @staticmethod
    def make_val_sparse_vector(drca_ls):
        index_array = []
        value_array = []
        #format is (#of columns, list(non-zero IDs), list(non-zero values))
        index = 0
        for val in drca_ls:
            if index == 0:
                index += 1
                continue
            if val != 0:
                index_array.append(index)
                value_array.append(drca_ls[index])
            index += 1
        sparse_vector = LabeledPoint(drca_ls[0], SparseVector(index, index_array, value_array))
        return sparse_vector

    @staticmethod
    def make_test_sparse_vector(drca_ls):
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
            index += 1
        sparse_vector = SparseVector(index, index_array, value_array)
        return sparse_vector

    #test code to find number of unique values in each column.
    #drop if the values are huge.
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

    #wrapper function of the whole data cleaning functions.
    @staticmethod
    def custom_function(ls, is_val, is_test):
        #One-Hot encoding
        a_ls = DataExploration.add_columns(ls)
        #Manipulate values
        rca_ls = DataExploration.replace_columns(a_ls)
        #Convert into numeric and target column to binary
        n_ls = HandleMissing.convert_into_numeric_value(rca_ls,
                                                        dict=DataExploration.header_dict,
                                                        birds_index=DataExploration.birds_column_ids,
                                                        drop_index=DataExploration.drop_column_ids)
        tn_ls = None
        #for testing
        if (not is_test):
            tn_ls = HandleMissing.convert_target_column_into_numeric(n_ls,
                                                                     target_index=DataExploration.get_col_id(
                                                                         "Agelaius_phoeniceus"), )
        else:
            tn_ls = n_ls
        #drop columns
        drca_ls = DataExploration.drop_columns(tn_ls)
        if is_val:
            sparse_vector = DataExploration.make_val_sparse_vector(drca_ls)
        elif is_test:
            sparse_vector = DataExploration.make_test_sparse_vector(drca_ls)
        else:
            sparse_vector = DataExploration.make_sparse_vector(drca_ls)
        return sparse_vector

    #create Column Name-ID mapping
    @staticmethod
    def create_header(headers):
        DataExploration.header_dict = DataExploration.create_header_dict(headers)

    def split_input_data(self, input_path, output_path):
        #splits the input data with seed to provide pseudo-randomness.
        seed = 17
        labelled = self.sc.textFile(input_path)
        headers = labelled.first()
        DataExploration.header_dict = DataExploration.create_header_dict(headers.split(","))
        print headers.split(",")
        modified_labels = labelled.subtract(self.sc.parallelize(headers))
        train, validate = modified_labels.randomSplit([9, 1], seed)
        train1, sample = validate.randomSplit([9, 1], seed)
        sample.saveAsTextFile(output_path)

    #wrapper to read function.
    def read_sample_training(self, file_path):
        return self.sc.textFile(file_path)

    #calculating correlation to decide feature importance.
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

    #populate the list of IDs belongs to Bird data
    @staticmethod
    def cal_birds_column_ids():
        bird_index = []
        for key, value in DataExploration.header_dict.items():
            parts = key.split('_')
            if(len(parts) > 1 and parts[1].islower()):
                bird_index.append(value)
        DataExploration.birds_column_ids = np.array(bird_index)

    #populate teh list of columns that are dropped.
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
    def replicate_data(value, nbr_of_models):
        duplication = []
        for i in range(nbr_of_models.value):
            duplication.append((i, value))
        return duplication

    def sparse_test(self, srdd):
        sprdd = srdd.filter(lambda x: DataExploration.filter_value_by_checklist_header(x))\
            .map(lambda x: DataExploration.swap_target(x)).\
            map(lambda x: DataExploration.custom_function(x, False, False))
        print sprdd.collect()[0]

    # Applying Machine Learning model to the data belonged to particular Group ID.
    # Here row is SparseVector which we convert into scipy spar matrix format
    # We training different machine learning model based upon the key
    @staticmethod
    def train_model(row):
        print "****", row
        labels = []
        features = []
        rows = []
        columns = []
        values = []
        total_column = 0
        total_row = 0
        key = row[0]
        #key = 0
        for train_data in row[1]:
        #for data in train_data[1]:
            #print train_data
            #key = train_data[0]
            label = train_data[0]
            total_column= train_data[1]
            actual_data = train_data[2]
            labels.append(label)
            for column_value in actual_data:
                column = column_value[0]
                value = column_value[1]
                rows.append(total_row)
                columns.append(column)
                values.append(value)
            total_row += 1
        '''
        for data in train_data[1]:
            labels.append(data[0])
            total_column = data[1]
            actual_data = data[2]
            for column_value in actual_data:
                column = column_value[0]
                value = column_value[1]
                rows.append(total_row)
                columns.append(column)
                values.append(value)
            total_row += 1
        '''
            #labels.append(data.label)
            #features.append(data.features.toArray())
            #features.append(data.features)
            # sps_acc = sps_acc + sps.coo_matrix((d, (r, c)), shape=(rows, cols))
        print '******************** total_row', total_row
        features = sp.csc_matrix((values, (rows, columns)), shape=(total_row, total_column))
        labels = np.array(labels)

        if int(key) == 0:
            return ModelTraining.train_sklean_neural_network(labels, features)
        elif int(key) == 1:
            return ModelTraining.train_sklean_random_forest(labels, features)
        elif int(key) == 2:
            return ModelTraining.train_sklean_gradient_trees(labels, features)
        elif int(key) == 3:
            return ModelTraining.train_sklean_logistic_regression(labels, features)
        elif int(key) == 4:
            return ModelTraining.train_sklean_adaboost(labels, features)
        else:
            return ModelTraining.train_sklean_logistic_regression(labels, features)
        #return [1]

    # Same as above method just the input type if different as the
    # this function is called directly after group by call.
    @staticmethod
    def train_model_after_group_by(train_data):
        labels = []
        rows = []
        columns = []
        values = []
        total_column = 0
        total_row = 0
        key = 0
        #print train_data
        for data in train_data:
            #print "data : ",data
            # for data in train_data[1]:
            key = data[0]
            actual_data = data[1]
            labels.append(actual_data[0])
            total_column = actual_data[1]
            feature_values = actual_data[2]
            for column_value in feature_values:
                column = column_value[0]
                value = column_value[1]
                rows.append(total_row)
                columns.append(column)
                values.append(value)
            total_row += 1
            # labels.append(data.label)
            # features.append(data.features.toArray())
            # features.append(data.features)
            # sps_acc = sps_acc + sps.coo_matrix((d, (r, c)), shape=(rows, cols))
        print '******************** total_row', total_row
        features = sp.csc_matrix((values, (rows, columns)), shape=(total_row, total_column))
        labels = np.array(labels)
        # features = np.array(features)

        if int(key) == 0:
            return ModelTraining.train_sklean_neural_network(labels, features)
        elif int(key) == 1:
            return ModelTraining.train_sklean_random_forest(labels, features)
        elif int(key) == 2:
            return ModelTraining.train_sklean_gradient_trees(labels, features)
        elif int(key) == 3:
            return ModelTraining.train_sklean_logistic_regression(labels, features)
        elif int(key) == 4:
            return ModelTraining.train_sklean_adaboost(labels, features)
        else:
            return ModelTraining.train_sklean_logistic_regression(labels, features)
            # return [1]

    # Train Machine Learing model on the given data
    def perform_distributed_ml(self, train_rdd, model_path):

        # Apply data cleaning and sampling operation
        processed_train_rdd = (train_rdd.filter(lambda x: DataExploration.filter_value_by_checklist_header(x)). \
                               map(lambda x: DataExploration.swap_target(x)). \
                               filter(lambda x: ModelTraining.handle_class_imbalance(x)). \
                               map(lambda x: DataExploration.custom_function(x, False, False)))

        print "Actual Count : " + str(processed_train_rdd.count())
        # Total Number of Machine Learning model to be trained on given dataset
        nbr_of_models = self.sc.broadcast(3)
        # Duplicate the input data and assign key from 0 to 2
        replicated_train_rdd = processed_train_rdd.flatMap(lambda x: DataExploration.replicate_data(x, nbr_of_models))#.keyBy(lambda x : x[0])
        print "Replicated Count : " + str(replicated_train_rdd.count())
        print replicated_train_rdd.first()

        # Apply group by so for each key value we can apply machine learning algorithm
        trained_group_by = replicated_train_rdd.groupBy(lambda x: x[0], numPartitions=3)
        #trained_group_by.coalesce(3, shuffle=True)
        models = trained_group_by.mapValues(lambda x: DataExploration.train_model_after_group_by(x))
        #models = trained_group_by.zipWithIndex().map(lambda x : (x[1], x[0])).mapValues(lambda x: DataExploration.train_model(x))
        print "Models : ", models.collect()
        models.saveAsPickleFile(model_path)

if __name__ == "__main__":

    dataExploration = DataExploration()

    args = sys.argv
    input_path = args[1] # "/Users/Darshan/Documents/MapReduce/FinalProject/LabeledSample"
    model_path = args[2] # "/Users/Darshan/Documents/MapReduce/FinalProject/Model"

    full_data_set = dataExploration.read_sample_training(input_path).persist()
    first_row = full_data_set.first().split(',')

    DataExploration.create_header(first_row)
    DataExploration.cal_birds_column_ids()
    DataExploration.cal_drop_column_ids()

    (train_rdd, val_rdd) = full_data_set.randomSplit([0.8, 0.2], 345)
    dataExploration.perform_distributed_ml(train_rdd, model_path)


