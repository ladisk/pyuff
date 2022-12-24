#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import sys, os
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + "/../")

import pyuff
from pyuff import tools

geom_exp_path = "./data/mesh_set2412_to_convert.uff"
file_geom = pyuff.UFF(geom_exp_path)
sets_old = file_geom.read_sets()
sets_new = tools.convert_dataset_2412_to_82(sets_old)

#check manually few values; better testing required
np.testing.assert_equal(sets_new[0]['type'], 82)
np.testing.assert_equal(sets_new[0]['trace_num'], 2)
np.testing.assert_equal(sets_new[0]['n_nodes'], 864)
np.testing.assert_equal(sets_new[0]['nodes'][:4], np.array([1,25,0,25]))