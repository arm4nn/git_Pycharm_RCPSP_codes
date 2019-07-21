import numpy as np
from gurobipy import *


class Constraints:

    @staticmethod
    def add_vars(grb, data, model):
        # one if activity i complete before activity j starts
        x = grb.addVars(data.activities, data.activities, lb=0.0, ub=1.0, vtype='B', name="X")
        # amount of resource r that will pass to activity j after the completion of activity i
        f = grb.addVars(data.activities, data.activities, data.resources, lb=0.0, ub=data.big_r, vtype='C', name="F")
        if model.type in {'fct'}:
            # start time of activity i
            z = grb.addVars(data.activities, lb=0.0, ub=data.big_t, vtype='C', name="Z")
            return x, z, f
        elif model.type in {'sfct', 'isfct'}:
            # start time of activity i in scenario s
            z = grb.addVars(data.activities, data.scenarios, lb=0.0, ub=data.big_t, vtype='C', name="Z")

            if model.type in {'sfct'}:
                return x, z, f
            elif model.type in {'isfct'}:
                # delivery date of procured materials/equipment for activity i
                d = grb.addVars(data.activities, lb=0.0, ub=data.big_t, vtype='C', name="O")
                return x, z, f, d
        else:
            raise Exception("model_name should be in {'fct', 'sfct', 'isfct'}")

    @staticmethod
    def add_const_1(grb, data, model):
        grb.addConstrs(
            (model.x[i, j] == 1
             for i in data.activities[:-1]
             for j in data.activities[data.successors[i][data.successors[i] > 0].astype(np.int) - 1]),
            name="Network Relations")

    @staticmethod
    def add_const_2a(grb, data, model):
        if model.type in {'sfct', 'isfct'}:
            grb.addConstrs(
                (model.z[j, s] - model.z[i, s] >= data.p_scn[i][s] - data.big_t * (1 - model.x[i, j])
                 for s in data.samples
                 for i in data.activities[:-1]
                 for j in data.activities[1:]),
                name="NetworkStartTimeRelations")
        elif model.type in {'fct'}:
            grb.addConstrs(
                (model.z[j] - model.z[i] >= data.duration[i] - data.big_t * (1 - model.x[i, j])
                 for i in data.activities[:-1]
                 for j in data.activities[1:]),
                name="NetworkStartTimeRelations")
        else:
            raise Exception("model_name should be in {'fct', 'sfct', 'isfct'}")

    @staticmethod
    def add_const_2b(grb, data, model):
        if model.type in {'isfct'}:
            grb.addConstrs(
                (model.z[i, s] >= model.d[i]
                 for s in data.samples
                 for i in data.activities),
                name="NetworkStartTimeAndOrderRelations")

    @staticmethod
    def add_const_3(grb, data, model):
        grb.addConstrs(
            (model.f[i, j, r] - data.big_r * model.x[i, j] <= 0
             for i in data.activities[:-1]
             for j in data.activities[1:]
             for r in data.resources),
            name="NetworkFlowRelations")

    @staticmethod
    def add_const_4(grb, data, model):
        grb.addConstrs(
            (quicksum(model.f[i, j, r] for j in data.activities[1:]) == data.res_use[i][r]
             for i in data.activities[1:]
             for r in data.resources),
            name="OutgoingFlows")

    @staticmethod
    def add_const_5(grb, data, model):
        grb.addConstrs(
            (quicksum(model.f[i, j, r] for i in data.activities[:-1]) == data.res_use[j][r]
             for j in data.activities[:-1]
             for r in data.resources),
            name="IngoingFlows")

    @staticmethod
    def add_const_6(grb, data, model):
        grb.addConstrs(
            (quicksum(model.f[data.activities[0], j, r] for j in data.activities[1:]) == data.available_resources[r]
             for r in data.resources),
            name="FirstFlow")

    @staticmethod
    def add_const_7(grb, data, model):
        grb.addConstrs(
            (quicksum(model.f[i, data.activities[-1], r] for i in data.activities[:-1]) == data.available_resources[r]
             for r in data.resources),
            name="LastFlow")

    @staticmethod
    def add_obj(grb, data, model):
        if model.type in {'fct'}:
            obj = (model.z[data.activities[-1]])
        elif model.type in {'sfct'}:
            obj = (quicksum(model.z[data.activities[-1], s] for s in data.samples) / data.sample_size)
        elif model.type in {'isfct'}:
            tmp_1 = quicksum(model.z[data.activities[-1], s] for s in data.samples) / data.sample_size
            tmp_2 = quicksum(model.w[i] * (model.z[i, s] - model.d[i]) for i in data.activities for s in
                             data.samples) / data.sample_size
            obj = tmp_1 + model.gamma * tmp_2
        else:
            raise Exception("model_name should be in {'fct', 'sfct', 'isfct'}")
        grb.setObjective(obj, GRB.MINIMIZE)
