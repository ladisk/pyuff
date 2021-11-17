#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import sys, os

my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + "/../")

import pyuff
from pyuff.datasets.convert_dataset_2412_to_82 import convert_dataset_2412_to_82

geom_exp_path = "C:/Pyuff/pyuff/data/mesh_set2412_to_convert.unv"
file_geom = pyuff.UFF(geom_exp_path)
sets_old = file_geom.read_sets()
file_new = convert_dataset_2412_to_82(file_geom)
