import numpy as np
import pandas as pd


class Data:
    def __init__(self, test_num, iteration=None, gamma=0.01):
        """
        :param test_num: (<1:48>,<1:10>)
        """
        self.test_num = test_num
        self.activities = None
        self.successors = None
        self.resources = None
        self.scenarios = None
        self.samples = None
        self.available_resources = None
        self.big_t = None  # upper bound on the duration of a project
        self.big_r = None  # upper bound on the amount of resources
        self.duration = None  # duration of each activity
        self.res_use = None  # resource usage of each activity
        self.scn_count = None  # number of scenarios
        self.sample_size = None  # sample size
        self.df = self.read_data()
        self.p_scn = None
        self.p_sample = None
        self.gamma = gamma
        self.w = self.gen_inventory_weights()
        self.iteration = iteration

    def read_data(self):
        file_name = 'J30' + str(self.test_num[0]) + '_' + str(self.test_num[1]) + '.RCP'
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
        self.big_t = 250
        self.available_resources = df.iloc[1, 0:resources_count].values
        self.big_r = self.available_resources.max()
        return df

    def gen_scn(self, sample_size, scn_count):
        self.scn_count = scn_count
        self.sample_size = sample_size
        self.scenarios = np.array(range(self.scn_count))
        self.samples = np.array(range(self.sample_size))
        if self.iteration is None:
            np.random.seed(5000 + self.test_num[0] * 10000 + self.test_num[1] * 100 + 1000000)
        else:
            np.random.seed(5000 + self.test_num[0] * 10000 + self.test_num[1] * 100 + self.iteration)
        dur_scn = np.zeros(shape=(32, self.scn_count))
        dur_sample = np.zeros(shape=(32, self.sample_size))
        

        #  based on our data, a good guess for the distribution of activities...
        #  ...is Weibull distribution with coefficient of variation 0.22
        for act in self.activities:
            for scn in self.scenarios:
                dur_scn[act, scn] = np.random.weibull(1.5) * 0.36 * self.duration[act] + 0.675 * self.duration[act]
            for scn in self.samples:
                dur_sample[act, scn] = np.random.weibull(1.5) * 0.36 * self.duration[act] + 0.675 * self.duration[act]

        self.p_scn = dict(zip(self.activities, dur_scn))
        self.p_sample = dict(zip(self.activities, dur_sample))
        
        for i in range(32):
            self.p_scn[i] = self.p_scn[i] + self.duration[i] - self.p_scn[i].mean()

    def gen_inventory_weights(self):
        np.random.seed(31475382 + self.test_num[0] * 10000 + self.test_num[1] * 100)
        w = [1 * (np.random.rand() > 0.5) * np.random.rand() for _ in self.activities]
        return w
