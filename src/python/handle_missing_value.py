from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.stat import Statistics
import numpy as np
import math
import random
import csv
from data_exploration import DataExploration

class HandleMissing:

    rem_ids = []

    @staticmethod
    def convert_target_into_numeric_value(values):
        target_index = values[DataExploration.get_col_id("Agelaius_phoeniceus")]
        try:
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
        target_index = values[DataExploration.get_col_id("Agelaius_phoeniceus")]
        value = values[target_index]
        if value == 0.0:
            return 0.0
        else:
            return 1.0

    @staticmethod
    def convert_birds_into_numeric_value(values, birds_index):
        for index in birds_index:
            if index != DataExploration.get_col_id("Agelaius_phoeniceus"):
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
    def convert_remaining_into_numeric_value(values, birds_index):
        if len(HandleMissing.rem_ids) == 0:
            remaining_index = []
            for ids in DataExploration.header_dict.values():
                try:
                    remaining_index.extend(ids)
                except TypeError:
                    remaining_index.append(ids)
            for id in remaining_index:
                if id in birds_index:
                    remaining_index.remove(id)
            HandleMissing.rem_ids = remaining_index

        for index in HandleMissing.rem_ids:
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
    def convert_into_numeric_value(values):
        tf_values = HandleMissing.convert_target_into_numeric_value(values)
        tfb_values = HandleMissing.convert_target_into_binary_value(tf_values)
        bv_values = HandleMissing.convert_birds_into_numeric_value(tfb_values)
        tv = HandleMissing.convert_remaining_into_numeric_value(bv_values)
        return tv

    @staticmethod
    def get_target_value(values):
        target_index = values[DataExploration.get_col_id("Agelaius_phoeniceus")]
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
