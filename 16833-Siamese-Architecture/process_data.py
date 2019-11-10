
import pandas as pd
import numpy as np
import os

from os import listdir
from os.path import isfile, join
import pandas as pd

PATH = 'daVinci/test/image_0/'
filesnames = [f for f in listdir(PATH) if isfile(join(PATH, f))]


df = pd.DataFrame(filesnames)

df.to_csv('test.csv')
