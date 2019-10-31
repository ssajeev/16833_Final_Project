# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 20:24:47 2019

@author: Nancy
"""
import pandas as pd
import numpy as np
import os

from os import listdir
from os.path import isfile, join
import pandas as pd

PATH1 = 'H:/16833/daVinci/test/image_0/'
filesnames_l = [join(PATH1, f) for f in listdir(PATH1) if isfile(join(PATH1, f))]
PATH2 = 'H:/16833/daVinci/test/image_1/'
filesnames_r = [join(PATH2, f) for f in listdir(PATH2) if isfile(join(PATH2, f))]

filename = 'endoscopy.txt'
file = open(filename ,'w')
for i in range(len(filesnames_l)):
    file.writelines(filesnames_l[i]+" "+filesnames_r[i]+"\n")