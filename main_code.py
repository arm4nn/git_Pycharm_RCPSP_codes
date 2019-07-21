import numpy as np
import pandas as pd
# from gurobipy import *


class Data:
    def __init__(self, test_num):
        """
        :param test_num: (<1:48>,<1:10>)
        """
        file_name = 'J30' + str(test_num[0]) + '_' + str(test_num[1]) + '.RCP'
        file = './data/A30/j30rcp/' + file_name
        columns = ["0", "1", "2", "3", "4", "5", "6", "7", "8"]
        df = pd.read_table(file, sep='\t', delim_whitespace=True, engine='python', names=columns)

        self.duration = df.iloc[2:, 0].values
        self.resources_usage = df.iloc[2:, 1:5].values
        self.successors_count = df.iloc[2:, 5].values
        self.successors = df.iloc[2:, 6:9].values
        self.resources_count = int(df.iloc[0, 1])
        self.jobs_count = int(df.iloc[0, 0])
        self.available_resources = df.iloc[1, 0:self.resources_count].values
