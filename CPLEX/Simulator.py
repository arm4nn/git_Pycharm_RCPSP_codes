import numpy as np
from gurobipy import *
from Data import Data


class Simulator:
    def __init__(self, model_type='fct', file=None, data=None):
        """
        :param test_num: (<1:48>,<1:10>)
        :param model_type: 'fct', 'sfct', or 'isfct'
        """
        # self.data = Data(test_num)
        # self.data.gen_scn(sample_size, scn_count)
        self.data = data
        self.grb = Model('model: Simulator')
        if model_type not in {'fct', 'sfct', 'isfct'}:
            raise Exception("model_name should be in {'fct', 'sfct', 'isfct'}")
        self.type = model_type
        self.x = np.loadtxt(file).astype(int)
        self.z = None
        self.f = None
        self.d = None
        self.obj = None
        self.det_x = np.zeros(1024).reshape(32, 32)
        self.time = None
        self.gap = None
        self.status = False

        self.solve()

    def solve(self):
        self.add_vars()
        self.add_const_2a()
        self.add_const_2b()
        self.add_const_3()
        self.add_const_4()
        self.add_const_5()
        self.add_const_6()
        self.add_const_7()
        self.add_obj()
        self.grb.setParam('TimeLimit', 60)
        self.grb.setParam('OutputFlag', 0)
        self.grb.optimize()
        if self.grb.status == 2:
            self.status = True
            self.obj = self.grb.ObjVal
            self.time = self.grb.Runtime
        else:
            print("******** problem {} is not solved optimally.".format(self.type))

    def add_vars(self):
        # amount of resource r that will pass to activity j after the completion of activity i
        self.f = self.grb.addVars(self.data.activities, self.data.activities, self.data.resources, lb=0.0,
                                  ub=self.data.big_r, vtype='C', name="F")
        if self.type in {'fct'}:
            # start time of activity i
            self.z = self.grb.addVars(self.data.activities, lb=0.0, ub=self.data.big_t, vtype='C', name="Z")
        elif self.type in {'sfct', 'isfct'}:
            # start time of activity i in scenario s
            self.z = self.grb.addVars(self.data.activities, self.data.scenarios, lb=0.0, ub=self.data.big_t, vtype='C',
                                      name="Z")
            if self.type in {'isfct'}:
                # delivery date of procured materials/equipment for activity i
                self.d = self.grb.addVars(self.data.activities, lb=0.0, ub=self.data.big_t, vtype='C', name="O")
        else:
            raise Exception("model_name should be in {'fct', 'sfct', 'isfct'}")


    def add_const_2a(self):
        if self.type in {'sfct', 'isfct'}:
            for i in self.data.activities[:-1]:
                for j in self.data.activities[1:]:
                    if self.x[i, j] == 1:
                        self.grb.addConstrs(
                            (self.z[j, s] - self.z[i, s] >= self.data.p_scn[i][s]
                             for s in self.data.scenarios),
                            name="NetworkStartTimeRelations")
        elif self.type in {'fct'}:
            self.grb.addConstrs(
                (self.z[j] - self.z[i] >= self.data.duration[i] - self.data.big_t * (1 - self.x[i, j])
                 for i in self.data.activities[:-1]
                 for j in self.data.activities[1:]),
                name="NetworkStartTimeRelations")
        else:
            raise Exception("model_name should be in {'fct', 'sfct', 'isfct'}")


    def add_const_2b(self):
        if self.type in {'isfct'}:
            self.grb.addConstrs(
                (self.z[i, s] >= self.d[i]
                 for s in self.data.scenarios
                 for i in self.data.activities),
                name="NetworkStartTimeAndOrderRelations")

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
        if self.type in {'fct'}:
            obj = (self.z[self.data.activities[-1]])
        elif self.type in {'sfct'}:
            obj = (quicksum(self.z[self.data.activities[-1], s] for s in self.data.scenarios) / self.data.scn_count)
        elif self.type in {'isfct'}:
            tmp_1 = quicksum(self.z[self.data.activities[-1], s] for s in self.data.scenarios) / self.data.scn_count
            tmp_2 = quicksum(self.data.w[i] * (self.d[i] - self.z[i, s]) for i in self.data.activities for s in
                             self.data.scenarios) / self.data.scn_count
            obj = tmp_1 + self.data.gamma * tmp_2
        else:
            raise Exception("model_name should be in {'fct', 'sfct', 'isfct'}")
        self.grb.setObjective(obj, GRB.MINIMIZE)


