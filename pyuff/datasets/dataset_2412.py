import numpy as np
import itertools

from ..tools import _opt_fields, _parse_header_line, check_dict_for_none

# FE descriptor ids of beam elements (rod, linear, tapered, curved and parabolic
# beam); their record 2 holds the orientation node and the cross sections
BEAM_DESCRIPTORS = {11, 21, 22, 23, 24}

def get_structure_2412(raw=False):
    """(source: https://www.ceas3.uc.edu/sdrluff/"""
    out = """
Universal Dataset Number: 2412

Name:   Elements
-----------------------------------------------------------------------
 
Record 1:        FORMAT(6I10)
                 Field 1       -- element label
                 Field 2       -- fe descriptor id
                 Field 3       -- physical property table number
                 Field 4       -- material property table number
                 Field 5       -- color
                 Field 6       -- number of nodes on element
 
Record 2:  *** FOR NON-BEAM ELEMENTS ***
                 FORMAT(8I10)
                 Fields 1-n    -- node labels defining element
 
Record 2:  *** FOR BEAM ELEMENTS ONLY ***
                 FORMAT(3I10)
                 Field 1       -- beam orientation node number
                 Field 2       -- beam fore-end cross section number
                 Field 3       -- beam  aft-end cross section number
 
Record 3:  *** FOR BEAM ELEMENTS ONLY ***
                 FORMAT(8I10)
                 Fields 1-n    -- node labels defining element
 
Records 1 and 2 are repeated for each non-beam element in the model.
Records 1 - 3 are repeated for each beam element in the model.
 
------------------------------------------------------------------------------
"""

    if raw:
        return out
    else:
        print(out)   

def _write2412(fh, dset):
    try:
        fh.write('%6i\n%6i%74s\n' % (-1, 2412, ' '))
        for el_type in dset:
            if type(el_type) is not int:
                # Skip 'type', 'triangle' and 'quad' indices
                continue
            for elem in dset[el_type]:
                fh.write('%10i%10i%10i%10i%10i%10i\n' % (
                    elem['element_nums'],
                    elem['fe_descriptor'],
                    elem['phys_table'],
                    elem['mat_table'],
                    elem['color'],
                    elem['num_nodes'],
                ))
                if elem['fe_descriptor'] in BEAM_DESCRIPTORS:
                    # Beam elements have an additional line before the nodes
                    fh.write('%10i%10i%10i\n' % (
                        elem['beam_orientation'],
                        elem['beam_foreend_cross'],
                        elem['beam_aftend_cross']
                    ))
                # Node labels, at most 8 in a line (format 8I10)
                nodes = list(elem['nodes_nums'])
                for j in range(0, len(nodes), 8):
                    fh.write('%10i' * len(nodes[j:j + 8]) % tuple(nodes[j:j + 8]) + '\n')
        fh.write('%6i\n' % -1)
    except:
        raise Exception('Error writing data-set #2412')


def _read_node_labels(split_data, i, num_nodes):
    """Returns the node labels of an element, which start at line i, and the
    index of the line after them.

    The labels are in lines of at most 8 (format 8I10); files written by
    pyuff 2.6.1 and older have all labels in one line.
    """
    nodes = [int(e) for e in split_data[i]]
    i += 1
    while len(nodes) < num_nodes:
        nodes += [int(e) for e in split_data[i]]
        i += 1
    return nodes, i


def _extract2412(block_data):
    """Extract element data - data-set 2412."""
    dset = {'type': 2412}
    # Define dictionary of possible elements types for legacy interface
    elt_type_dict = {41: 'triangle', 44: 'quad'}

    # Read data
    try:
        split_data = block_data.splitlines()
        split_data = [a.split() for a in split_data][2:]

        # Extract Records
        i = 0
        while i < len(split_data):
            dict_tmp = dict()
            line = split_data[i]
            dict_tmp['element_nums'] = int(line[0])
            dict_tmp['fe_descriptor'] = int(line[1])
            dict_tmp['phys_table'] = int(line[2])
            dict_tmp['mat_table'] = int(line[3])
            dict_tmp['color'] = int(line[4])
            dict_tmp['num_nodes'] = int(line[5])
            if dict_tmp['fe_descriptor'] in BEAM_DESCRIPTORS:
                # element is a rod and covers 3 lines
                dict_tmp['beam_orientation'] = int(split_data[i+1][0])
                dict_tmp['beam_foreend_cross'] = int(split_data[i + 1][1])
                dict_tmp['beam_aftend_cross'] = int(split_data[i + 1][2])
                dict_tmp['nodes_nums'], i = _read_node_labels(split_data, i + 2, dict_tmp['num_nodes'])
            else:
                # element is no rod: record 1 is followed by the node labels
                dict_tmp['nodes_nums'], i = _read_node_labels(split_data, i + 1, dict_tmp['num_nodes'])
            desc = dict_tmp['fe_descriptor']
            if not desc in dset:
                dset[desc] = []
            dset[desc].append(dict_tmp)
        for num, name in elt_type_dict.items():
            # if we have one of the keys that are enabled for the legacy interface, add everything for that here
            if num in dset.keys():
                dset[name] = {
                    'element_nums': [e['element_nums'] for e in dset[num]],
                    'fe_descriptor': [e['fe_descriptor'] for e in dset[num]],
                    'phys_table': [e['phys_table'] for e in dset[num]],
                    'mat_table': [e['mat_table'] for e in dset[num]],
                    'color': [e['color'] for e in dset[num]],
                    'num_nodes': [e['num_nodes'] for e in dset[num]],
                    'nodes_nums': [e['nodes_nums'] for e in dset[num]],
                }

        return dset

    except:
        raise Exception('Error reading data-set #2412')


def prepare_2412(
        element_nums=None,
        fe_descriptor=None,
        phys_table=None,
        mat_table=None,
        color=None,
        num_nodes=None,
        nodes_nums=None,
        beam_orientation=None,
        beam_foreend_cross=None,
        beam_aftend_cross=None,
        return_full_dict=False):
    """Prepare one element of data-set 2412 (elements) for writing.

    Data-set 2412 holds the finite elements of a model; in modal testing it is used to draw the
    mesh that connects the nodes of data-set 2411 (lines and surfaces between the measurement
    points). Each element has a label, an FE descriptor id (the element type), property and
    material table numbers, a color and the labels of its nodes.

    This function prepares a single element. The data-set to write is a dictionary with the key
    ``'type'`` set to 2412 and, for each FE descriptor id in the model, an integer key with the
    list of the elements of that type, e.g. ``{'type': 2412, 21: [beam], 44: [quad1, quad2]}``.
    :meth:`pyuff.UFF.read_sets` returns data-set 2412 in the same form; it additionally gives the
    elements with descriptors 41 and 44 under the keys ``'triangle'`` and ``'quad'``, which are
    ignored when writing.

    Beam elements (FE descriptor ids 11, 21, 22, 23 and 24) have an additional record with the
    beam orientation node and the cross section numbers at the two ends of the beam
    (``beam_orientation``, ``beam_foreend_cross`` and ``beam_aftend_cross``); they are required
    for beam elements and ignored for the other elements. The node labels are written 8 per line.

    Parameters
    ----------
    element_nums : int
        Element label (record 1, field 1).
    fe_descriptor : int
        FE descriptor id, the element type (record 1, field 2). Common ids: 11 rod, 21 linear
        beam, 22 tapered beam, 23 curved beam, 24 parabolic beam, 41 plane stress linear
        triangle, 42 plane stress parabolic triangle, 44 plane stress linear quadrilateral,
        45 plane stress parabolic quadrilateral, 91 thin shell linear triangle, 92 thin shell
        parabolic triangle, 94 thin shell linear quadrilateral, 95 thin shell parabolic
        quadrilateral, 111 solid linear tetrahedron, 112 solid linear wedge, 115 solid linear
        brick, 116 solid parabolic brick, 118 solid parabolic tetrahedron, 122 rigid element,
        136 node-to-node translational spring, 161 lumped mass. The full list is in the
        specification at https://www.ceas3.uc.edu/sdrluff/.
    phys_table : int
        Physical property table number, e.g. of the cross section or thickness; 0 if not used
        (record 1, field 3).
    mat_table : int
        Material property table number; 0 if not used (record 1, field 4).
    color : int
        Color number, used by pre- and post-processors only (record 1, field 5).
    num_nodes : int
        Number of nodes of the element; must equal the length of ``nodes_nums``, it is not
        computed when writing (record 1, field 6).
    nodes_nums : array_like of int
        Labels of the nodes that define the element, in the order of the element type, length
        ``num_nodes`` (record 2 for non-beam elements, record 3 for beam elements; 8 labels per
        line).
    beam_orientation : int, optional
        Beam orientation node number, the node that defines the orientation of the cross
        section; 0 if not used (record 2, field 1). Required for beam elements.
    beam_foreend_cross : int, optional
        Cross section number at the fore end (end A, first node) of the beam (record 2,
        field 2). Required for beam elements.
    beam_aftend_cross : int, optional
        Cross section number at the aft end (end B, last node) of the beam (record 2, field 3).
        Required for beam elements.
    return_full_dict : bool, optional
        If True, the dictionary has all keys of the data-set, with ``None`` for
        the values not given; if False (default), only the given values.

    Returns
    -------
    dict
        One element. Put it in the list under its FE descriptor id in the data-set dictionary
        (see above), which is written with :meth:`pyuff.UFF.write_sets`.

    Examples
    --------
    A beam and two quadrilateral shell elements:

    >>> import pyuff
    >>> beam = pyuff.prepare_2412(
    ...     element_nums=1, fe_descriptor=21, phys_table=1, mat_table=1, color=8,
    ...     num_nodes=2, nodes_nums=[1, 2],
    ...     beam_orientation=0, beam_foreend_cross=1, beam_aftend_cross=1)
    >>> quads = [
    ...     pyuff.prepare_2412(
    ...         element_nums=2, fe_descriptor=94, phys_table=2, mat_table=1, color=8,
    ...         num_nodes=4, nodes_nums=[1, 2, 5, 4]),
    ...     pyuff.prepare_2412(
    ...         element_nums=3, fe_descriptor=94, phys_table=2, mat_table=1, color=8,
    ...         num_nodes=4, nodes_nums=[2, 3, 6, 5])]
    >>> elements = {'type': 2412, 21: [beam], 94: quads}
    >>> pyuff.UFF('elements.uff').write_sets(elements, mode='overwrite')
    """
    if np.array(element_nums).dtype != int and element_nums != None:
        raise TypeError('element_nums must be integer')
    if np.array(fe_descriptor).dtype != int and fe_descriptor != None:
        raise TypeError('fe_descriptor must be integer')
    if np.array(phys_table).dtype != int and phys_table != None:
        raise TypeError('phys_table must be integer')
    if np.array(mat_table).dtype != int and mat_table != None:
        raise TypeError('mat_table must be integer')
    if np.array(color).dtype != int and color != None:
        raise TypeError('color must be integer')
    if np.array(num_nodes).dtype != int and num_nodes != None:
        raise TypeError('num_nodes must be integer')
    if np.array(nodes_nums).dtype != int and nodes_nums != None:
        raise TypeError('nodes_nums must be integer')
    if np.array(beam_orientation).dtype != int and beam_orientation != None:
        raise TypeError('beam_orientation must be integer')
    if np.array(beam_foreend_cross).dtype != int and beam_foreend_cross != None:
        raise TypeError('beam_foreend_cross must be integer')
    if np.array(beam_aftend_cross).dtype != int and beam_aftend_cross != None:
        raise TypeError('beam_aftend_cross must be integer')


    dataset={
        'type': 2412,
        'element_nums': element_nums,
        'fe_descriptor': fe_descriptor,
        'phys_table': phys_table,
        'mat_table': mat_table,
        'color': color,
        'num_nodes': num_nodes,
        'beam_orientation': beam_orientation,
        'beam_foreend_cross': beam_foreend_cross,
        'beam_aftend_cross': beam_aftend_cross,
        'nodes_nums': nodes_nums
        }
    
    if return_full_dict is False:
        dataset = check_dict_for_none(dataset)

    return dataset

