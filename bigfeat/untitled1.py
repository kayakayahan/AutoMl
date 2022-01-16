# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 23:08:23 2021

@author: kayak
"""

import bigfeat_base 
import pandas as pd

data= pd.read_csv('housing.csv')

y=data['price']

data=data.drop('price',axis=1)

print(data.columns)

x=bigfeat_base.BigFeat()

x.fit(data,y)







