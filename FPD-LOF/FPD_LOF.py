#intersections = ['196003','201212','201225','201231','201234','201239','201245','201249','201291','201297','201302','201308','201311','205250']

# missing: '205229'
# problem: 196003
#import pandas as pd
#import numpy as np
#from dictances import bhattacharyya as bha
#from sklearn.neighbors import LocalOutlierFactor as LOF
#import math
#import os
#import pickle
#import json
#import copy

import Packages
all_results = Packages.create_historical_outlier_dfs()

import pickle
pickle.dump(all_results, open("results.pickle","wb"))