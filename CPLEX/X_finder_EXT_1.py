import numpy as np
from gurobipy import *
from Data import Data


class X_finder_EXT_1:
    def __init__(self, test_num=(1,1), sample_size=20, scn_count=2000, iteration=0, MIPGap=0.0001, dic={}):
        """
        :param test_num: (<1:48>,<1:10>)
        """
        self.data = Data(test_num, iteration)
        self.data.gen_scn(sample_size, scn_count)
        self.grb = Model('model: X finder')
        self.MIPGap = MIPGap
        self.x = None
        self.z = None
        self.f = None
        self.d = None
        self.obj = None
        self.det_x = np.zeros(1024).reshape(32, 32)
        self.time = None
        self.gap = None
        self.status = False
        self.dic = dic[test_num]
        self.makespan = dic[test_num][32][1]


        self.solve()

    def solve(self):
        self.add_vars()
        self.add_const_1()
        self.add_const_2()
        self.add_const_3()
        self.add_const_4()
        self.add_const_5()
        self.add_const_6()
        self.add_const_7()
        self.add_obj()
        self.grb.setParam('TimeLimit', 600)
        self.grb.setParam('OutputFlag', 1)
        self.grb.setParam('MIPGap', self.MIPGap)
        self.grb.update()
        self.grb.optimize()
        if self.grb.status == 2:
            self.status = True
            self.obj = self.grb.ObjVal
            self.time = self.grb.Runtime
            self.gap = self.grb.MIPGap
            if len(self.x) == 1024:
                for i in range(32):
                    for j in range(32):
                        self.det_x[i, j] = round(self.x[i, j].X)
            else:
                raise Exception("len(self.x) != 1024")
        else:
            print("******** problem is not solved optimally.")

    def add_vars(self):
        # one if activity i complete before activity j starts
        self.x = self.grb.addVars(self.data.activities, self.data.activities, lb=0.0, ub=1.0, vtype='B', name="X")
        # amount of resource r that will pass to activity j after the completion of activity i
        self.f = self.grb.addVars(self.data.activities, self.data.activities, self.data.resources, lb=0.0,
                                  ub=self.data.big_r, vtype='C', name="F")
        # z is a parameter in this model
        self.z = [self.dic[k][0] for k in self.dic]
        
    def add_const_1(self):
        self.grb.addConstrs(
            (self.x[i, j] >= self.x[i, k] + self.x[k, j] - 1
             for i in self.data.activities
             for j in self.data.activities
             for k in self.data.activities),
            name="EXT_1")
        
    def add_const_1(self):
        self.grb.addConstrs(
            (self.x[i, j] == 1
             for i in self.data.activities[:-1]
             for j in self.data.activities[self.data.successors[i][self.data.successors[i] > 0].astype(np.int) - 1]),
            name="Network Relations")

    def add_const_2(self):
        self.grb.addConstrs(
            (self.z[j] - self.z[i] >= self.data.duration[i] - self.data.big_t * (1 - self.x[i, j])
             for i in self.data.activities[:-1]
             for j in self.data.activities[1:]),
            name="NetworkStartTimeRelations")

    def add_const_3(self):
        self.grb.addConstrs(
            (self.f[i, j, r] - self.data.big_r * self.x[i, j] <= 0
             for i in self.data.activities[:-1]
             for j in self.data.activities[1:]
             for r in self.data.resources),
            name="NetworkFlowRelations")

    def add_const_4(self):
        self.grb.addConstrs(
            (quicksum(self.f[i, j, r] for j in self.data.activities[1:]) == self.data.res_use[i][r]
             for i in self.data.activities[1:]
             for r in self.data.resources),
            name="OutgoingFlows")

    def add_const_5(self):
        self.grb.addConstrs(
            (quicksum(self.f[i, j, r] for i in self.data.activities[:-1]) == self.data.res_use[j][r]
             for j in self.data.activities[:-1]
             for r in self.data.resources),
            name="IngoingFlows")

    def add_const_6(self):
        self.grb.addConstrs(
            (quicksum(self.f[self.data.activities[0], j, r] for j in self.data.activities[1:]) ==
             self.data.available_resources[r]
             for r in self.data.resources),
            name="FirstFlow")

    def add_const_7(self):
        self.grb.addConstrs(
            (quicksum(self.f[i, self.data.activities[-1], r] for i in self.data.activities[:-1]) ==
             self.data.available_resources[r]
             for r in self.data.resources),
            name="LastFlow")

    def add_obj(self):
        obj = quicksum(self.x[i,j] for i in self.data.activities for j in self.data.activities)
        self.grb.setObjective(obj, GRB.MINIMIZE)

