#region
"""
Created on 04/15/2024

@author: Dan
"""

#endregion
#region # IMPORTS
# =============================================================================
# IMPORTS
# =============================================================================
import os
os.getcwd()

import numpy as np
import pandas as pd
import pycaret
import numpy as np
import matplotlib.pyplot as plt
from pycaret.classification import * 

#endregion
#region # LOAD DATA
# =============================================================================
# LOAD DATA
# =============================================================================
train = pd.read_csv('./data/fundraising.csv')
for item in train.columns:
    print(item)
test = pd.read_csv('./data/future_fundraising.csv')
test['target'] = np.nan
comboed = pd.concat([train, test], ignore_index=True)

#endregion
#region # CLEAN DATA
# =============================================================================
# CLEAN DATA
# =============================================================================
# train.head()
# train.columns
# train['zipconvert2'].unique()
# # convert yes / nos to 1 0s
#     # zip converts
#     # homeowner
#     # female
#     # traget
# comboed.columns
# columns_to_convert = [
#     'zipconvert2',
#     'zipconvert3',
#     'zipconvert4',
#     'zipconvert5',
#     'homeowner',
#     'female',
#     'target'
#      ]

# # Loop through columns and convert yes/no to 1/0
# for column in columns_to_convert:
#     train[column] = np.where(train[column] == 'yes', 1, 0)
#     test[column] = np.where(test[column] == 'yes', 1, 0)
#     comboed[column] = np.where(comboed[column] == 'yes', 1, 0)
# target == target

#endregion
#region # PYCARET 
# =============================================================================
# PYCARET 
# =============================================================================
s = setup(train, target = 'target', session_id = 27)

from pycaret.classification import ClassificationExperiment
exp = ClassificationExperiment()
exp.setup(train, target = 'target', session_id = 27)

best = compare_models()