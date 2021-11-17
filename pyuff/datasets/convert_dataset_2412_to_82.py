import pyuff
import numpy as np
from ..datasets.dataset_82 import _write82
from ..datasets.dataset_15 import _write15


def convert_dataset_2412_to_82(file_path):

    """
    Parameters :
    ---------------------------------------
    file_path : path to write new file with set 82 instead of set 2412

    Return:
    --------------------------------------
    New file with set 82 instead of set 2412
    """

    ###read sets from uff initial file
    datasets = file_path.read_sets()
    ###create new file name
    file_path_new = file_path._filename[:-4] + "_new.unv"
    file_new = pyuff.UFF(file_path_new)

    ###Initialize list of nodes to add to dataset 82
    nodes = list()
    for i in range(0, len(datasets)):
        if (datasets[i]["type"] == 2412) and ("quad" in datasets[i].keys()):
            for j in range(len(datasets[i]["quad"]["nodes_nums"])):
                nodes.append(datasets[i]["quad"]["nodes_nums"][j][0])
                nodes.append(datasets[i]["quad"]["nodes_nums"][j][1])
                nodes.append(0)
                nodes.append(datasets[i]["quad"]["nodes_nums"][j][1])
                nodes.append(datasets[i]["quad"]["nodes_nums"][j][2])
                nodes.append(0)
                nodes.append(datasets[i]["quad"]["nodes_nums"][j][2])
                nodes.append(datasets[i]["quad"]["nodes_nums"][j][3])
                nodes.append(0)
                nodes.append(datasets[i]["quad"]["nodes_nums"][j][3])
                nodes.append(datasets[i]["quad"]["nodes_nums"][j][0])
                nodes.append(0)
            nodes = np.array(nodes)
            dataset_82 = pyuff.prepare_82(
                trace_num=i,
                n_nodes=len(nodes),
                nodes=nodes,
                color=0,
                id="Stator",
                return_full_dict=True,
            )
            file_new._write_set(dataset_82, "add")
        else:
            file_new._write_set(datasets[i], "add")

    return file_new