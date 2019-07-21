import numpy as np
import pandas as pd
from gurobipy import *


class Data:
    def __init__(self, test_num):
        """
        :param test_num: (<1:48>,<1:10>)
        """
        self.activities = None
        self.successors = None
        self.resources = None
        self.scenarios = None
        self.samples = None
        self.big_t = None  # upper bound on the duration of a project
        self.big_r = None  # upper bound on the amount of resources
        self.duration = None  # duration of each activity
        self.res_use = None  # resource usage of each activity
        self.scn_count = None  # number of scenarios
        self.sample_size = None  # sample size
        self.df = self.read_data(test_num)
        self.p_scn = None

    def read_data(self, test_num):
        file_name = 'J30' + str(test_num[0]) + '_' + str(test_num[1]) + '.RCP'
        file = './data/A30/j30rcp/' + file_name
        columns = ["0", "1", "2", "3", "4", "5", "6", "7", "8"]
        df = pd.read_csv(file, delim_whitespace=True, engine='python', names=columns)
        self.activities = np.array(range(32))
        self.duration = df.iloc[2:, 0].values
        resources_usage = df.iloc[2:, 1:5].values
        self.successors = df.iloc[2:, 6:9].values
        resources_count = int(df.iloc[0, 1])
        self.resources = np.array(range(resources_count))
        self.res_use = dict(zip(self.activities, resources_usage))
        self.big_t = 150
        available_resources = df.iloc[1, 0:resources_count].values
        self.big_r = available_resources.max()
        return df

    def gen_scn(self, scn_count, sample_size):
        self.scn_count = scn_count
        self.sample_size = sample_size
        self.scenarios = np.array(range(self.scn_count))
        self.samples = np.array(range(self.sample_size))
        np.random.seed(5000)
        rnd_list = np.random.rand(32)
        duration_scn = np.zeros(shape=(32, self.scn_count))
        for scn in self.scenarios:
            for act in self.activities:
                if np.random.rand() >= 0.5:
                    duration_scn[act, scn] = self.duration[act] * (1 + 5 / 13 * rnd_list[act])
                else:
                    duration_scn[act, scn] = self.duration[act] * (1 - 5 / 13 * rnd_list[act])
        self.p_scn = dict(zip(self.activities, duration_scn))


X = Data((10, 10))
print(X.df)
