import numpy as np
import pandas as pd
from gurobipy import *
import bokeh

bokeh.output_notebook()

scnCount = 3000
subScnCount = 20

Number = [7]
SubNumber = [1]
directory = "./11/"


# In[3]:


# reading data
def read_data(readNumber, readSubNumber):
    jobs = 30
    test_number = readNumber  # 1 to 48
    subTestNumber = readSubNumber  # 1 to 10
    fileName = 'J' + str(jobs) + str(test_number) + '_' + str(subTestNumber) + '.RCP'
    file = '../dataRCPSP/A' + str(jobs) + '/j' + str(jobs) + 'rcp/' + fileName
    my_cols = ["0", "1", "2", "3", "4", "5", "6", "7", "8"]
    df = pd.read_table(file, sep='\t', delim_whitespace=True, engine='python', names=my_cols)

    duration = df.iloc[2:, 0].values
    resourcesUsage = df.iloc[2:, 1:5].values
    successorsCount = df.iloc[2:, 5].values
    successors = df.iloc[2:, 6:9].values
    resourcesCount = int(df.iloc[0, 1])
    jobsCount = int(df.iloc[0, 0])
    availableResources = df.iloc[1, 0:resourcesCount].values
    return duration, resourcesUsage, successors, resourcesCount, jobsCount, availableResources


# In[4]:


# model sets and parameters for Solver model
def FCTSetParam(jobsCount, resourcesCount, duration, resourcesUsage, availableResources, scenarioCount,
                subScenarioCount):
    activities = np.array(range(jobsCount))
    resources = np.array(range(resourcesCount))
    scenarios = np.array(range(scenarioCount))
    subScenarios = np.array(range(subScenarioCount))
    T = duration.sum()
    d = dict(zip(activities, resourcesUsage))
    M = 150
    N = availableResources.max()
    return activities, resources, d, T, M, N, scenarios, subScenarios


# In[5]:


# scenario generating
def scenarioGen(scenarios, jobsCount, duration, activities):
    np.random.seed(0)
    rndAct = np.random.rand(jobsCount)
    durationScenarios = np.zeros(shape=(jobsCount, scenarios[-1] + 1))
    for s in scenarios:
        for a in activities:
            cr = np.random.rand()
            if (cr >= 0.5):
                durationScenarios[a, s] = duration[a] * (1 + 5 / 13 * rndAct[a])
            elif (cr < 0.5):
                durationScenarios[a, s] = duration[a] * (1 - 5 / 13 * rndAct[a])
    p = dict(zip(activities, durationScenarios))
    return p


# In[6]:


# define variables of Solver model
def addFCTVars(activities, resources, T, availableResources):
    x = FCT.addVars(activities, activities, lb=0.0, ub=1.0, vtype='B',
                    name="X")  # one if activity i complete before activity j starts
    s = FCT.addVars(activities, scenarios, lb=0.0, ub=T, vtype='C',
                    name="S")  # start time of activity i in scenario s
    o = FCT.addVars(activities, lb=0.0, ub=T, vtype='C',
                    name="O")  # due date of procurements for activity i
    e = FCT.addVars(activities, scenarios, lb=0.0, ub=T, vtype='C',
                    name="E")  # earliness of materials for activity i in scenario s
    f = FCT.addVars(activities, activities, resources, lb=0.0, ub=availableResources.max(), vtype='C',
                    name="F")  # amount of resource r that after completion of activity i will pass to activity j

    return x, s, f, o, e


# In[7]:


# define variables of Solver model
def addFCTVarsMM(activities, resources, T, availableResources):
    x = FCT.addVars(activities, activities, lb=0.0, ub=1.0, vtype='B',
                    name="X")  # one if activity i complete before activity j starts
    s = FCT.addVars(activities, lb=0.0, ub=T, vtype='C',
                    name="S")  # start time of activity i in scenario s
    f = FCT.addVars(activities, activities, resources, lb=0.0, ub=availableResources.max(), vtype='C',
                    name="F")  # amount of resource r that after completion of activity i will pass to activity j

    return x, s, f


# In[8]:


# define variables of Solver model
def addFCTVarsXX(activities, resources, T, availableResources):
    s = FCT.addVars(activities, scenarios, lb=0.0, ub=T, vtype='C',
                    name="S")  # start time of activity i in scenario s
    o = FCT.addVars(activities, lb=0.0, ub=T, vtype='C',
                    name="O")  # due date of procurements for activity i
    e = FCT.addVars(activities, scenarios, lb=0.0, ub=T, vtype='C',
                    name="E")  # earliness of materials for activity i in scenario s
    f = FCT.addVars(activities, activities, resources, lb=0.0, ub=availableResources.max(), vtype='C',
                    name="F")  # amount of resource r that after completion of activity i will pass to activity j

    return s, f, o, e


# In[9]:


# constraint 2.5 of Solver model
def addFCTConst1(activities, successors, x):
    FCT.addConstrs(
        (x[activityI, activityJ]
         ==
         1
         for activityI in activities if activityI != activities[-1]
         for activityJ in activities[successors[activityI][successors[activityI] > 0].astype(np.int) - 1]),
        name="NetworkRelations")


# In[10]:


# constraint 2.6a of Solver model
def addFCTConst2a(activities, x, s, p, M, scenarios):
    FCT.addConstrs(
        (s[activityJ, scenario] - s[activityI, scenario]
         >=
         p[activityI][scenario] - M * (1 - x[activityI, activityJ])
         for scenario in scenarios
         for activityI in activities if activityI != activities[-1]
         for activityJ in activities if activityJ != activities[0]),
        name="NetworkStartTimeRelations")


# In[11]:


# constraint 2.6a of Solver model
def addFCTConst2aMM(activities, x, s, ppp, M):
    FCT.addConstrs(
        (s[activityJ] - s[activityI]
         >=
         ppp[activityI] - M * (1 - x[activityI, activityJ])
         for activityI in activities if activityI != activities[-1]
         for activityJ in activities if activityJ != activities[0]),
        name="NetworkStartTimeRelations")


# In[12]:


# constraint 2.6b of Solver model
def addFCTConst2b(activities, s, o, scenarios):
    FCT.addConstrs(
        (s[activity, scenario]
         >=
         o[activity]
         for scenario in scenarios
         for activity in activities),
        name="NetworkStartTimeAndOrderRelations")


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[8]:


t = 'ali2'
a = [''] * len(t)

for i, e in enumerate(t):
    a[len(t) - i - 1] = e

''.join(a)

# In[ ]:


df = pd.read_table(file, sep='\t', delim_whitespace=True, engine='python', names=my_cols)

# In[21]:


'hello world'[::-1]

# In[24]:


reversed(text)

# In[23]:


text = 'adatatg'


# In[ ]:


# In[ ]:


# In[ ]:


# In[13]:


# constraint 2.7 of Solver model
def addFCTConst3(activities, x, f, resources, N):
    FCT.addConstrs(
        (f[activityI, activityJ, resource] - N * x[activityI, activityJ]
         <=
         0
         for activityI in activities if activityI != activities[-1]
         for activityJ in activities if activityJ != activities[0]
         for resource in resources),
        name="NetworkFlowRelations")


# In[14]:


# constraint 2.8 of Solver model
def addFCTConst4(activities, f, resources, d):
    FCT.addConstrs(
        (quicksum(f[activityI, activityJ, resource]
                  for activityJ in activities
                  if activityJ != activities[0])
         ==
         d[activityI][resource]
         for activityI in activities if activityI != activities[0]
         for resource in resources),
        name="OutgoingFlows")


# In[15]:


# constraint 2.9 of Solver model
def addFCTConst5(activities, f, resources, d):
    FCT.addConstrs(
        (quicksum(f[activityI, activityJ, resource]
                  for activityI in activities
                  if activityI != activities[-1])
         ==
         d[activityJ][resource]
         for activityJ in activities if activityJ != activities[-1]
         for resource in resources),
        name="IngoingFlows")


# In[16]:


# constraint 2.10a of Solver model
def addFCTConst6(activities, f, resources, availableResources):
    FCT.addConstrs(
        (quicksum(f[activities[0], activityJ, resource]
                  for activityJ in activities
                  if activityJ != activities[0])
         ==
         availableResources[resource]
         for resource in resources),
        name="FirstFlow")


# In[17]:


# constraint 2.10b of Solver model
def addFCTConst7(activities, f, resources, availableResources):
    FCT.addConstrs(
        (quicksum(f[activityI, activities[-1], resource]
                  for activityI in activities
                  if activityI != activities[-1])
         ==
         availableResources[resource]
         for resource in resources),
        name="LastFlow")


# In[18]:


# constraint for bounding makespan
def addFCTConst9(activities, scenarios, scenarioCount, s, makespan):
    FCT.addConstrs(
        (quicksum(s[activities[-1], scenario] for scenario in scenarios)
         <=
         makespan * scenarioCount
         for alaki in range(0, 1)
         ),
        name="Makespandddd")


# In[19]:


# Objective of Solver model -- min cost
def addFCTObj(activities, s, scenarios, scenarioCount, o):
    obj = (quicksum(s[activity, scenario] - o[activity]
                    for activity in activities
                    for scenario in scenarios) / scenarioCount)
    FCT.setObjective(obj, GRB.MINIMIZE)


# In[20]:


# Objective of Solver model -- min makespan det
def addFCTObjMM(activities, s):
    obj = (s[activities[-1]])
    FCT.setObjective(obj, GRB.MINIMIZE)


# In[21]:


# Objective of Solver model -- min expected makespan
def addFCTObjdW(activities, s, scenarios, scenarioCount):
    obj = (quicksum(s[activities[-1], scenario] for scenario in scenarios) / scenarioCount)
    FCT.setObjective(obj, GRB.MINIMIZE)


# In[22]:


for i in Number:
    for j in SubNumber:

        # MM##########################################################################################################
        ################################# MIN makespan for DETERMINISTIC
        print("iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii_" + str(i) + "_" + str(j))
        print("    MIN makespan for DETERMINISTIC    step=1  ")
        print("iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii")
        ############################################################################################################1

        makespanDet = 0
        makespanExp = 0

        duration, resourcesUsage, successors, resourcesCount, jobsCount, availableResources = read_data(i, j)
        activities, resources, d, T, M, N, scenarios, subScenarios = FCTSetParam(jobsCount, resourcesCount, duration,
                                                                                 resourcesUsage, availableResources,
                                                                                 scnCount, subScnCount)
        p = scenarioGen(scenarios, jobsCount, duration, activities)
        pMM = list(range(jobsCount))
        for act in range(jobsCount):
            pMM[act] = sum(p[act]) / scnCount
        FCT = Model('Solver: MIN makespan for DETERMINISTIC')
        # Solver.setParam('TimeLimit', 3600)
        x, s, f = addFCTVarsMM(activities, resources, T, availableResources)
        addFCTConst1(activities, successors, x)
        addFCTConst2aMM(activities, x, s, pMM, M)
        addFCTConst3(activities, x, f, resources, N)
        addFCTConst4(activities, f, resources, d)
        addFCTConst5(activities, f, resources, d)
        addFCTConst6(activities, f, resources, availableResources)
        addFCTConst7(activities, f, resources, availableResources)
        FCT.update()
        addFCTObjMM(activities, s)
        FCT.optimize()

        # .............................................. Save MM-X

        # directory = directory + str(i) + "o" + str(j) + "o/"
        # !mkdir $directory
        if FCT.Status == 2:

            # Open a file X
            xoName = directory + "MM-x.txt"
            xo = open(xoName, "a")
            for v in range(jobsCount):
                if v != 0: xo.write("\n")
                for k in range(jobsCount):
                    xo.write(str(int(x[v, k].X)) + "\t")
            # Close opend file X
            xo.close()

        # .............................................. Update eff.txt

        preName = "|A" + str(jobsCount - 2) + "|" + str(i) + "_" + str(j) + "|." + str(FCT.Status) + ".|MM|\t\t"
        if FCT.Status == 2:
            # Open a file
            do = open(directory + "eff.txt", "a")
            do.write(preName)
            do.write(str(round(FCT.ObjVal, 4)) + "\t\t" + str(round(FCT.Runtime, 1)) + " sec\n")
            # Close opend file
            do.close()
        else:
            # Open a file
            do = open(directory + "eff.txt", "a")
            do.write(preName)
            do.write("Unable to find the optimum solution in " + str(round(FCT.Runtime, 1)) + " sec\n")
            # Close opend file
            do.close()

        # dW############################################################################################################
        ##################### MIN E(makespan) for SCENARIO base | XdW bede
        print("iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii_" + str(i) + "_" + str(j))
        print("    MIN E(makespan) for SCENARIO base | XdW bede   step=2 ")
        print("iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii")
        ##############################################################################################################2

        subP = scenarioGen(subScenarios, jobsCount, duration, activities)
        FCT = Model('Solver: MIN makespan for SCENARIO base')
        # Solver.setParam('TimeLimit', 3600)
        x, s, f, o, e = addFCTVars(activities, resources, T, availableResources)
        addFCTConst1(activities, successors, x)
        addFCTConst2a(activities, x, s, subP, M, subScenarios)
        addFCTConst2b(activities, s, o, subScenarios)
        addFCTConst3(activities, x, f, resources, N)
        addFCTConst4(activities, f, resources, d)
        addFCTConst5(activities, f, resources, d)
        addFCTConst6(activities, f, resources, availableResources)
        addFCTConst7(activities, f, resources, availableResources)
        FCT.update()
        addFCTObjdW(activities, s, subScenarios, subScnCount)
        FCT.optimize()

        # .............................................. Save dW-X

        if FCT.Status == 2:
            # Open a file X
            xoName = directory + "dW-x.txt"
            xo = open(xoName, "a")
            for v in range(jobsCount):
                if v != 0: xo.write("\n")
                for k in range(jobsCount):
                    xo.write(str(int(x[v, k].X)) + "\t")
            # Close opend file X
            xo.close()

        # .............................................. Update eff.txt

        # Calculations
        if FCT.MIPGap <= 10000:
            aaa = 0  # Expected Makespan
            for vvv in range(subScnCount):
                aaa += s[jobsCount - 1, vvv].X
        preName = "|A" + str(jobsCount - 2) + "|" + str(i) + "_" + str(j) + "|." + str(FCT.Status) + ".|dW|\t\t"
        if FCT.Status == 2:
            # Open a file
            do = open(directory + "eff.txt", "a")
            do.write(preName)
            do.write(str(round(aaa / subScnCount, 4)) + "\t\t" + str(round(FCT.Runtime, 1)) + " sec\n")
            # Close opend file
            do.close()
        else:
            # Open a file
            do = open(directory + "eff.txt", "a")
            do.write(preName)
            do.write("Unable to find the optimum solution in " + str(round(FCT.Runtime, 1)) + " sec\n")
            # Close opend file
            do.close()

        # SdW###############################################################################################################
        ####################################   for fixed X: MIN E(makespan) | use x of dW shema bede
        print("iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii_" + str(i) + "_" + str(j))
        print("  for fixed X: MIN E(makespan) | use x of dW shema bede   step=3 ")
        print("iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii")
        #################################################################################################################3

        fileName = "dW-x.txt"
        file = directory + fileName
        x = np.loadtxt(file)
        x = x.astype(int)

        FCT = Model('for fixed X: MIN scenario used makespan | use x of dW')
        # Solver.setParam('TimeLimit', 10*60)
        s, f, o, e = addFCTVarsXX(activities, resources, T, availableResources)
        #             addFCTConst1(activities, successors, x)
        addFCTConst2a(activities, x, s, p, M, scenarios)
        addFCTConst2b(activities, s, o, scenarios)
        addFCTConst3(activities, x, f, resources, N)
        addFCTConst4(activities, f, resources, d)
        addFCTConst5(activities, f, resources, d)
        addFCTConst6(activities, f, resources, availableResources)
        addFCTConst7(activities, f, resources, availableResources)
        #         addFCTConst9(activities, scenarios, scnCount, s, makespan)
        FCT.update()
        addFCTObjdW(activities, s, scenarios, scnCount)
        FCT.optimize()

        # .............................................. Update eff.txt

        # Calculations
        preName = "|A" + str(jobsCount - 2) + "|" + str(i) + "_" + str(j) + "|." + str(FCT.Status) + ".|SdW|\t\t"
        if FCT.Status == 2:
            # Open a file
            makespanExp = FCT.ObjVal
            do = open(directory + "eff.txt", "a")
            do.write(preName)
            do.write(str(round(FCT.ObjVal, 4)) + "\t\t" + str(round(FCT.Runtime, 1)) + " sec\n")
            # Close opend file
            do.close()
        else:
            # Open a file
            do = open(directory + "eff.txt", "a")
            do.write(preName)
            do.write("Unable to find the optimum solution in " + str(round(FCT.Runtime, 1)) + " sec\n")
            # Close opend file
            do.close()

        # SMM###############################################################################################################
        ####################################   for fixed X: MIN E(makespan) | use x of MM shema bede
        print("iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii_" + str(i) + "_" + str(j))
        print("  for fixed X: MIN E(makespan) | use x of MM shema bede   step=4 ")
        print("iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii")
        #################################################################################################################4

        fileName = "MM-x.txt"
        file = directory + fileName
        x = np.loadtxt(file)
        x = x.astype(int)

        FCT = Model('for fixed X: MIN scenario used makespan | use x of MM')
        # Solver.setParam('TimeLimit', 10*60)
        s, f, o, e = addFCTVarsXX(activities, resources, T, availableResources)
        #             addFCTConst1(activities, successors, x)
        addFCTConst2a(activities, x, s, p, M, scenarios)
        addFCTConst2b(activities, s, o, scenarios)
        addFCTConst3(activities, x, f, resources, N)
        addFCTConst4(activities, f, resources, d)
        addFCTConst5(activities, f, resources, d)
        addFCTConst6(activities, f, resources, availableResources)
        addFCTConst7(activities, f, resources, availableResources)
        #         addFCTConst9(activities, scenarios, scnCount, s, makespan)
        FCT.update()
        addFCTObjdW(activities, s, scenarios, scnCount)
        FCT.optimize()

        # .............................................. Update eff.txt

        # Calculations
        preName = "|A" + str(jobsCount - 2) + "|" + str(i) + "_" + str(j) + "|." + str(FCT.Status) + ".|SMM|\t\t"
        if FCT.Status == 2:
            # Open a file
            makespanDet = FCT.ObjVal
            do = open(directory + "eff.txt", "a")
            do.write(preName)
            do.write(str(round(FCT.ObjVal, 4)) + "\t\t" + str(round(FCT.Runtime, 1)) + " sec\n")
            # Close opend file
            do.close()
        else:
            # Open a file
            do = open(directory + "eff.txt", "a")
            do.write(preName)
            do.write("Unable to find the optimum solution in " + str(round(FCT.Runtime, 1)) + " sec\n")
            # Close opend file
            do.close()

        # XX##########################################################################################################
        ######################   for fixed X, bounded makespan: MIN cost | use x of MM
        print("iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii_" + str(i) + "_" + str(j))
        print("  for fixed X, bounded makespan: MIN cost | use x of MM   step=5_0 ")
        print("iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii")
        ############################################################################################################5

        fileName = "MM-x.txt"
        file = directory + fileName
        x = np.loadtxt(file)
        x = x.astype(int)

        makespansXX = makespanDet * np.array(
            [1, 1.001, 1.002, 1.003, 1.005, 1.0075, 1.01, 1.015, 1.02, 1.025, 1.03, 1.05, 1.1, 1.2])
        XX = {}
        XX1 = {}
        shomarande = 0
        for makespan in makespansXX:
            shomarande += 1
            print("iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii_" + str(i) + "_" + str(j))
            print("  for fixed X, bounded makespan: MIN cost | use x of MM   step=5: " + str(shomarande) + "/" + str(
                len(makespansXX)))
            print("iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii")

            FCT = Model('Solver: for fixed X, bounded makespan: MIN cost')
            # Solver.setParam('TimeLimit', 10*60)
            s, f, o, e = addFCTVarsXX(activities, resources, T, availableResources)
            #             addFCTConst1(activities, successors, x)
            addFCTConst2a(activities, x, s, p, M, scenarios)
            addFCTConst2b(activities, s, o, scenarios)
            addFCTConst3(activities, x, f, resources, N)
            addFCTConst4(activities, f, resources, d)
            addFCTConst5(activities, f, resources, d)
            addFCTConst6(activities, f, resources, availableResources)
            addFCTConst7(activities, f, resources, availableResources)
            addFCTConst9(activities, scenarios, scnCount, s, makespan)
            FCT.update()
            addFCTObj(activities, s, scenarios, scnCount, o)
            FCT.optimize()

            # .............................................. Update eff.txt

            # Calculations
            preName = "|A" + str(jobsCount - 2) + "|" + str(i) + "_" + str(j) + "|." + str(FCT.Status) + ".|XX|\t\t"
            if FCT.Status == 2:
                ccc = 0  # Worst Makespan
                for vvv in range(scnCount):
                    if ccc < s[jobsCount - 1, vvv].X:
                        ccc = s[jobsCount - 1, vvv].X
                # Open a file
                do = open(directory + "eff.txt", "a")
                do.write(preName)
                do.write(str(round(makespan, 4)) + "\t\t" + str(round(FCT.ObjVal, 4)) + "   \t" + str(
                    round(ccc, 4)) + "\t\t" + str(round(FCT.Runtime, 1)) + " sec\n")
                XX[makespan] = FCT.ObjVal

                qctmp = []
                for ac in activities:
                    for sc in scenarios:
                        qctmp.append(s[ac, sc].X)
                XX1[makespan] = sum(qctmp) / scnCount

                # Close opend file
                do.close()
            else:
                # Open a file
                do = open(directory + "eff.txt", "a")
                do.write(preName)
                do.write("Unable to find the optimum solution in " + str(round(FCT.Runtime, 1)) + " sec\n")
                # Close opend file
                do.close()

            # WW##############################################################################################################
        ###########################   for fixed X, bounded makespan: MIN cost | use x of dW
        print("iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii_" + str(i) + "_" + str(j))
        print("  for fixed X, bounded makespan: MIN cost | use x of dW   step=6_0 ")
        print("iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii")
        ################################################################################################################6

        fileName = "dW-x.txt"
        file = directory + fileName
        x = np.loadtxt(file)
        x = x.astype(int)

        mianji = 1 + (makespanDet - makespanExp) / (2 * makespanExp)

        makespansWW = np.hstack((makespanExp * np.array([1, mianji]), makespansXX))
        WW = {}
        WW1 = {}
        shomarande = 0
        for makespan in makespansWW:

            shomarande += 1
            print("iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii_" + str(i) + "_" + str(j))
            print("  for fixed X, bounded makespan: MIN cost | use x of dW   step=6: " + str(shomarande) + "/" + str(
                len(makespansWW)))
            print("iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii")

            FCT = Model('Solver: for fixed X, bounded makespan: MIN cost')
            # Solver.setParam('TimeLimit', 10*60)
            s, f, o, e = addFCTVarsXX(activities, resources, T, availableResources)
            #             addFCTConst1(activities, successors, x)
            addFCTConst2a(activities, x, s, p, M, scenarios)
            addFCTConst2b(activities, s, o, scenarios)
            addFCTConst3(activities, x, f, resources, N)
            addFCTConst4(activities, f, resources, d)
            addFCTConst5(activities, f, resources, d)
            addFCTConst6(activities, f, resources, availableResources)
            addFCTConst7(activities, f, resources, availableResources)
            addFCTConst9(activities, scenarios, scnCount, s, makespan)
            FCT.update()
            addFCTObj(activities, s, scenarios, scnCount, o)
            FCT.optimize()

            # .............................................. Update eff.txt

            # Calculations
            preName = "A" + str(jobsCount - 2) + "|" + str(i) + "_" + str(j) + "|." + str(FCT.Status) + ".|WW|\t\t"
            if FCT.Status == 2:
                ccc = 0  # Worst Makespan
                for vvv in range(scnCount):
                    if ccc < s[jobsCount - 1, vvv].X:
                        ccc = s[jobsCount - 1, vvv].X
                WW[makespan] = FCT.ObjVal

                qctmp = []
                for ac in activities:
                    for sc in scenarios:
                        qctmp.append(s[ac, sc].X)
                WW1[makespan] = sum(qctmp) / scnCount

                # Open a file
                do = open(directory + "eff.txt", "a")
                do.write(preName)
                do.write(str(round(makespan, 4)) + "\t\t" + str(round(FCT.ObjVal, 4)) + "   \t" + str(
                    round(ccc, 4)) + "\t\t" + str(round(FCT.Runtime, 1)) + " sec\n")
                # Close opend file
                do.close()
            else:
                # Open a file
                do = open(directory + "eff.txt", "a")
                do.write(preName)
                do.write("Unable to find the optimum solution in " + str(round(FCT.Runtime, 1)) + " sec\n")
                # Close opend file
                do.close()

            # FIGURE##########################################################################################################
        ############################################ FIGURE
        ################################################################################################################7

        vWW = np.array(list(WW.values()))
        kWW = np.array(list(WW.keys()))
        dfWW = pd.DataFrame({'kWW': kWW, 'vWW': vWW})
        sourceWW = ColumnDataSource(dfWW)

        vXX = np.array(list(XX.values()))
        kXX = np.array(list(XX.keys()))
        dfXX = pd.DataFrame({'kXX': kXX, 'vXX': vXX})
        sourceXX = ColumnDataSource(dfXX)

        # plotting

        plt = figure(plot_width=1200, plot_height=800, title="Efficient Frontier", tools='pan,wheel_zoom',
                     x_axis_label='Expected Completion Time', y_axis_label='Expected Inventory Cost')

        plt.line(x='kWW', y='vWW', source=sourceWW, line_width=10, alpha=.3, color='red')
        plt.circle(x='kWW', y='vWW', source=sourceWW, line_width=10, alpha=.3, legend="Heuristic", color='red',
                   hover_fill_color='firebrick', hover_alpha=1,
                   hover_line_color='white')

        plt.line(x='kXX', y='vXX', source=sourceXX, line_width=10, alpha=.3, color='black')
        plt.circle(x='kXX', y='vXX', source=sourceXX, line_width=10, alpha=.3, legend="Deterministic", color='black',
                   hover_fill_color='firebrick', hover_alpha=1,
                   hover_line_color='white')

        plt.legend.location = 'top_right'
        plt.legend.background_fill_color = 'floralwhite'

        hover = HoverTool(tooltips=None)
        plt.add_tools(hover)

        pltName = "ScnBased_" + str(i) + "_" + str(j) + ".html"
        output_file(pltName)
        show(plt)

        curdoc().clear()

# In[27]:


# FIGURE##########################################################################################################
############################################ FIGURE
################################################################################################################7

vWW = np.array(list(WW.values()))
kWW = np.array(list(WW.keys()))
dfWW = pd.DataFrame({'kWW': kWW, 'vWW': vWW})
sourceWW = ColumnDataSource(dfWW)

vXX = np.array(list(XX.values()))
kXX = np.array(list(XX.keys()))
dfXX = pd.DataFrame({'kXX': kXX, 'vXX': vXX})
sourceXX = ColumnDataSource(dfXX)

vXX1 = np.array(list(XX1.values()))
kXX1 = np.array(list(XX1.keys()))
dfXX1 = pd.DataFrame({'kXX1': kXX1, 'vXX1': vXX1})
sourceXX1 = ColumnDataSource(dfXX1)

# plotting

plt = figure(plot_width=1200, plot_height=800, title="Efficient Frontier", tools='pan,wheel_zoom,save',
             x_axis_label='Expected Completion Time', y_axis_label='Expected Inventory Cost')

plt.line(x='kWW', y='vWW', source=sourceWW, line_width=10, alpha=.3, color='red')
plt.circle(x='kWW', y='vWW', source=sourceWW, line_width=10, alpha=.3, legend="Heuristic", color='red',
           hover_fill_color='firebrick', hover_alpha=1,
           hover_line_color='white')

plt.line(x='kXX', y='vXX', source=sourceXX, line_width=10, alpha=.3, color='black')
plt.circle(x='kXX', y='vXX', source=sourceXX, line_width=10, alpha=.3, legend="Deterministic", color='black',
           hover_fill_color='firebrick', hover_alpha=1,
           hover_line_color='white')

plt.line(x='kXX1', y='vXX1', source=sourceXX1, line_width=10, alpha=.3, color='purple')
plt.circle(x='kXX1', y='vXX1', source=sourceXX1, line_width=10, alpha=.3, legend="Ordering too soon", color='purple',
           hover_fill_color='firebrick', hover_alpha=1,
           hover_line_color='white')

plt.legend.location = 'center_right'
plt.legend.background_fill_color = 'floralwhite'

hover = HoverTool(tooltips=None)
plt.add_tools(hover)

pltName = "Banafsh.html"
output_file(pltName)
show(plt)

curdoc().clear()

# In[26]:


# FIGURE##########################################################################################################
############################################ FIGURE
################################################################################################################7

vWW = np.array(list(WW.values()))
kWW = np.array(list(WW.keys()))
dfWW = pd.DataFrame({'kWW': kWW, 'vWW': vWW})
sourceWW = ColumnDataSource(dfWW)

vXX = np.array(list(XX.values()))
kXX = np.array(list(XX.keys()))
dfXX = pd.DataFrame({'kXX': kXX, 'vXX': vXX})
sourceXX = ColumnDataSource(dfXX)

vXX1 = np.array(list(XX1.values()))
kXX1 = np.array(list(XX1.keys()))
dfXX1 = pd.DataFrame({'kXX1': kXX1, 'vXX1': vXX1})
sourceXX1 = ColumnDataSource(dfXX1)

# plotting

plt = figure(plot_width=1200, plot_height=800, title="Efficient Frontier", tools='pan,wheel_zoom,save',
             x_axis_label='Expected Completion Time', y_axis_label='Expected Inventory Cost')

plt.line(x='kWW', y='vWW', source=sourceWW, line_width=10, alpha=.3, color='red')
plt.circle(x='kWW', y='vWW', source=sourceWW, line_width=10, alpha=.3, legend="Heuristic", color='red',
           hover_fill_color='firebrick', hover_alpha=1,
           hover_line_color='white')

plt.line(x='kXX', y='vXX', source=sourceXX, line_width=10, alpha=.3, color='black')
plt.circle(x='kXX', y='vXX', source=sourceXX, line_width=10, alpha=.3, legend="Deterministic", color='black',
           hover_fill_color='firebrick', hover_alpha=1,
           hover_line_color='white')

# plt.line(x='kXX1', y='vXX1', source=sourceXX1, line_width=10, alpha=.3, color='purple')
# plt.circle(x='kXX1', y='vXX1', source=sourceXX1, line_width=10, alpha=.3, legend="Ordering too soon", color='purple', hover_fill_color='firebrick', hover_alpha=1,
#           hover_line_color='white')


plt.legend.location = 'center_right'
plt.legend.background_fill_color = 'floralwhite'

hover = HoverTool(tooltips=None)
plt.add_tools(hover)

pltName = "BanafshNist.html"
output_file(pltName)
show(plt)

curdoc().clear()

# In[ ]:


# In[ ]:


# In[25]:


for i in Number:
    for j in SubNumber:

        # WW##############################################################################################################
        ###########################   for fixed X, bounded makespan: MIN cost | use x of dW
        print("iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii_" + str(i) + "_" + str(j))
        print("  for fixed X, bounded makespan: MIN cost | use x of dW   step=6_0 ")
        print("iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii")
        ################################################################################################################6

        fileName = "dW-x.txt"
        file = directory + fileName
        x = np.loadtxt(file)
        x = x.astype(int)

        mianji0 = 1 + (makespanDet - makespanExp) / makespanExp * 0.20
        mianji1 = 1 + (makespanDet - makespanExp) / makespanExp * 0.35
        mianji2 = 1 + (makespanDet - makespanExp) / makespanExp * 0.50
        mianji3 = 1 + (makespanDet - makespanExp) / makespanExp * 0.60
        mianji4 = 1 + (makespanDet - makespanExp) / makespanExp * 0.70
        mianji5 = 1 + (makespanDet - makespanExp) / makespanExp * 0.80
        mianji6 = 1 + (makespanDet - makespanExp) / makespanExp * 0.90

        makespansWW = np.hstack(
            (makespanExp * np.array([1, mianji0, mianji1, mianji2, mianji3, mianji4, mianji5, mianji6]), makespansXX))
        WW = {}
        WW1 = {}
        shomarande = 0
        for makespan in makespansWW:

            shomarande += 1
            print("iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii_" + str(i) + "_" + str(j))
            print("  for fixed X, bounded makespan: MIN cost | use x of dW   step=6: " + str(shomarande) + "/" + str(
                len(makespansWW)))
            print("iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii")

            FCT = Model('Solver: for fixed X, bounded makespan: MIN cost')
            # Solver.setParam('TimeLimit', 10*60)
            s, f, o, e = addFCTVarsXX(activities, resources, T, availableResources)
            #             addFCTConst1(activities, successors, x)
            addFCTConst2a(activities, x, s, p, M, scenarios)
            addFCTConst2b(activities, s, o, scenarios)
            addFCTConst3(activities, x, f, resources, N)
            addFCTConst4(activities, f, resources, d)
            addFCTConst5(activities, f, resources, d)
            addFCTConst6(activities, f, resources, availableResources)
            addFCTConst7(activities, f, resources, availableResources)
            addFCTConst9(activities, scenarios, scnCount, s, makespan)
            FCT.update()
            addFCTObj(activities, s, scenarios, scnCount, o)
            FCT.optimize()

            # .............................................. Update eff.txt

            # Calculations
            preName = "A" + str(jobsCount - 2) + "|" + str(i) + "_" + str(j) + "|." + str(FCT.Status) + ".|WW|\t\t"
            if FCT.Status == 2:
                ccc = 0  # Worst Makespan
                for vvv in range(scnCount):
                    if ccc < s[jobsCount - 1, vvv].X:
                        ccc = s[jobsCount - 1, vvv].X
                WW[makespan] = FCT.ObjVal

                qctmp = []
                for ac in activities:
                    for sc in scenarios:
                        qctmp.append(s[ac, sc].X)
                WW1[makespan] = sum(qctmp) / scnCount

                # Open a file
                do = open(directory + "eff.txt", "a")
                do.write(preName)
                do.write(str(round(makespan, 4)) + "\t\t" + str(round(FCT.ObjVal, 4)) + "   \t" + str(
                    round(ccc, 4)) + "\t\t" + str(round(FCT.Runtime, 1)) + " sec\n")
                # Close opend file
                do.close()
            else:
                # Open a file
                do = open(directory + "eff.txt", "a")
                do.write(preName)
                do.write("Unable to find the optimum solution in " + str(round(FCT.Runtime, 1)) + " sec\n")
                # Close opend file
                do.close()

            # FIGURE##########################################################################################################
        ############################################ FIGURE
        ################################################################################################################7

        vWW = np.array(list(WW.values()))
        kWW = np.array(list(WW.keys()))
        dfWW = pd.DataFrame({'kWW': kWW, 'vWW': vWW})
        sourceWW = ColumnDataSource(dfWW)

        vXX = np.array(list(XX.values()))
        kXX = np.array(list(XX.keys()))
        dfXX = pd.DataFrame({'kXX': kXX, 'vXX': vXX})
        sourceXX = ColumnDataSource(dfXX)

        # plotting

        plt = figure(plot_width=1200, plot_height=800, title="Efficient Frontier", tools='pan,wheel_zoom',
                     x_axis_label='Expected Completion Time', y_axis_label='Expected Inventory Cost')

        plt.line(x='kWW', y='vWW', source=sourceWW, line_width=10, alpha=.3, color='red')
        plt.circle(x='kWW', y='vWW', source=sourceWW, line_width=10, alpha=.3, legend="Heuristic", color='red',
                   hover_fill_color='firebrick', hover_alpha=1,
                   hover_line_color='white')

        plt.line(x='kXX', y='vXX', source=sourceXX, line_width=10, alpha=.3, color='black')
        plt.circle(x='kXX', y='vXX', source=sourceXX, line_width=10, alpha=.3, legend="Deterministic", color='black',
                   hover_fill_color='firebrick', hover_alpha=1,
                   hover_line_color='white')

        plt.legend.location = 'top_right'
        plt.legend.background_fill_color = 'floralwhite'

        hover = HoverTool(tooltips=None)
        plt.add_tools(hover)

        pltName = "ScnBased_" + str(i) + "_" + str(j) + ".html"
        output_file(pltName)
        show(plt)

        curdoc().clear()

# In[ ]:
