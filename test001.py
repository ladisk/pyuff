import numpy as np
import matplotlib.pyplot as plt
import pyuff

uff_file = pyuff.UFF('data/fea_simufact_tet_elements.unv')

datasets_in_uff = uff_file.get_set_types()
print('datasets in uff:')
print(datasets_in_uff)

# read and print all datasets
count = 0
for type in datasets_in_uff:
    print('---------------------------------------------------------------')
    print('dataset        : ' + str(count) )
    print('dataset type   : ' + str(type) )
    print('dataset content: ' )
    dataset = uff_file.read_sets(count)
    print(dataset)
    count += 1
    

