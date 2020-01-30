import numpy as np
from gurobipy import *
from Data import Data


class InventorySimulator:
    def __init__(self, file=None, data=None, bound=100.0):
        self.data = data
        self.grb = Model('model: Inventory Simulator')
        self.x = np.loadtxt(file).astype(int)
        self.z = None
        self.f = None
        self.d = None
        self.obj = None
        self.det_x = np.zeros(1024).reshape(32, 32)
        self.time = None
        self.gap = None
        self.status = False
        self.bound = bound

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
        self.add_const_8()
        self.add_obj()
        self.grb.setParam('TimeLimit', 120)
        self.grb.setParam('OutputFlag', 0)
        self.grb.optimize()
        if self.grb.status == 2:
            self.status = True
            self.obj = self.grb.ObjVal
            self.time = self.grb.Runtime
        else:
            print("******** problem {} is not solved optimally.".format(self.data.test_num))

    def add_vars(self):
        # amount of resource r that will pass to activity j after the completion of activity i
        self.f = self.grb.addVars(self.data.activities, self.data.activities, self.data.resources, lb=0.0,
                                  ub=self.data.big_r, vtype='C', name="F")

        # start time of activity i in scenario s
        self.z = self.grb.addVars(self.data.activities, self.data.scenarios, lb=0.0, ub=self.data.big_t, vtype='C',
                                  name="Z")
        # delivery date of procured materials/equipment for activity i
        self.d = self.grb.addVars(self.data.activities, lb=0.0, ub=self.data.big_t, vtype='C', name="O")

    def add_const_2a(self):
        for i in self.data.activities[:-1]:
            for j in self.data.activities[1:]:
                if self.x[i, j] == 1:
                    self.grb.addConstrs(
                        (self.z[j, s] - self.z[i, s] >= self.data.p_scn[i][s]
                         for s in self.data.scenarios),
                        name="NetworkStartTimeRelations")

    def add_const_2b(self):
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

    def add_const_8(self):
        self.grb.addConstr(
            (quicksum(
                self.z[self.data.activities[-1], s] for s in self.data.scenarios) / self.data.scn_count <= self.bound),
            name="ExpectedMakespanBound")

    def add_obj(self):
        obj = quicksum(self.data.w[i] * (self.z[i, s] - self.d[i]) for i in self.data.activities for s in
                       self.data.scenarios) / self.data.scn_count
        self.grb.setObjective(obj, GRB.MINIMIZE)
