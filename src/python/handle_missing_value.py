import random
import numpy as np


class HandleMissing:
    #Remaining IDs (Un-Dropped)
    rem_ids = []
    target_index = 0

    def __init__(self):
        self.rem_ids = []

    #bird columns - converted to numeric
    @staticmethod
    def convert_birds_into_numeric_value(values, birds_index):
        for index in birds_index:
            if index != HandleMissing.target_index:
                value = values[index]
                try:
                    #if the values is x , the observer forgot the count. So we replace it with random number between 2,10
                    if value == 'x':
                        values[index] = float(random.randint(2,10))
                    #? replaced with 0
                    elif value == '?':
                        values[index] = 0.0
                    else:
                        values[index] = float(value)
                except:
                    values[index] = 0.0
        return values

    #Same as above, separate function for loose coupling.
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
        #convert target into binary so we give it to ML algorithm
        value = values[HandleMissing.target_index]
        if value == 0.0:
            values[HandleMissing.target_index] = 0.0
        else:
            values[HandleMissing.target_index] = 1.0
        return values

    #convert the remaining columns (un-dropped and Non-bird columns)
    @staticmethod
    def convert_remaining_into_numeric_value(values, birds_index, drop_index, dict):
        #populate remaining IDs list ONLY ONCE.
        if len(HandleMissing.rem_ids) == 0:
            remaining_index = []
            #Some columns are list of IDs. Handle them accordingly
            #add every ID first
            for ids in dict.values():
                try:
                    remaining_index.extend(ids)
                except TypeError:
                    remaining_index.append(ids)
            #subtract the BIRD and Dropped IDs
            temp_remaining_index = []
            for id in remaining_index:
                if id not in birds_index and id not in drop_index:
                    temp_remaining_index.append(id)

            HandleMissing.rem_ids = temp_remaining_index
            HandleMissing.rem_ids.sort()

        for index in HandleMissing.rem_ids:
            try:
                value = values[index]
                #Same logic as above.
                if value == 'x':
                    values[index] = float(random.randint(2,10))
                elif value == '?':
                    values[index] = 0.0
                else:
                    values[index] = float(value)
            except:
                values[index] = 0.0
        return values

    #wrapper to convert columns to numeric
    @staticmethod
    def convert_into_numeric_value(values, dict, birds_index, drop_index):
        bv_values = HandleMissing.convert_birds_into_numeric_value(values, birds_index)
        tv = HandleMissing.convert_remaining_into_numeric_value(bv_values, birds_index, drop_index, dict)
        return tv

    #wrapper to convert target column to numeric and binary (Separate function for loose coupling)
    @staticmethod
    def convert_target_column_into_numeric(values, target_index):
        HandleMissing.target_index = target_index
        tf_values = HandleMissing.convert_target_into_numeric_value(values)
        tfb_values = HandleMissing.convert_target_into_binary_value(tf_values)
        return tfb_values
