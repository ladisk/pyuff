import warnings

import numpy as np

from ..tools import _opt_fields, _parse_header_line, check_dict_for_none, _fit_string

def get_structure_55(raw=False):
    """(source: https://www.ceas3.uc.edu/sdrluff/"""
    out = """
Universal Dataset Number: 55

Name:   Data at Nodes
-----------------------------------------------------------------------
 
          RECORD 1:      Format (40A2)
               FIELD 1:          ID Line 1
 
          RECORD 2:      Format (40A2)
               FIELD 1:          ID Line 2
 
          RECORD 3:      Format (40A2)
 
               FIELD 1:          ID Line 3
 
          RECORD 4:      Format (40A2)
               FIELD 1:          ID Line 4
 
          RECORD 5:      Format (40A2)
               FIELD 1:          ID Line 5
 
          RECORD 6:      Format (6I10)
 
          Data Definition Parameters
 
               FIELD 1: Model Type
                           0:   Unknown
                           1:   Structural
                           2:   Heat Transfer
                           3:   Fluid Flow
 
               FIELD 2:  Analysis Type
                           0:   Unknown
                           1:   Static
                           2:   Normal Mode
                           3:   Complex eigenvalue first order
                           4:   Transient
                           5:   Frequency Response
                           6:   Buckling
                           7:   Complex eigenvalue second order
 
               FIELD 3:  Data Characteristic
                           0:   Unknown
                           1:   Scalar
                           2:   3 DOF Global Translation 
                                Vector
                           3:   6 DOF Global Translation 
                                & Rotation Vector
                           4:   Symmetric Global Tensor
                           5:   General Global Tensor
 
               FIELD 4: Specific Data Type
                           0:   Unknown
                           1:   General
                           2:   Stress
                           3:   Strain
                           4:   Element Force
                           5:   Temperature
                           6:   Heat Flux
                           7:   Strain Energy
                           8:   Displacement
                           9:   Reaction Force
                           10:   Kinetic Energy
                           11:   Velocity
                           12:   Acceleration
                           13:   Strain Energy Density
                           14:   Kinetic Energy Density
                           15:   Hydro-Static Pressure
                           16:   Heat Gradient
                           17:   Code Checking Value
                           18:   Coefficient Of Pressure
 
               FIELD 5:  Data Type
                           2:   Real
                           5:   Complex
 
               FIELD 6:  Number Of Data Values Per Node (NDV)
 
 
          Records 7 And 8 Are Analysis Type Specific
 
          General Form
 
          RECORD 7:      Format (8I10)
 
               FIELD 1:          Number Of Integer Data Values
                           1 < Or = Nint < Or = 10
               FIELD 2:          Number Of Real Data Values
                           1 < Or = Nrval < Or = 12
               FIELDS 3-N:       Type Specific Integer Parameters
 
 
          RECORD 8:      Format (6E13.5)
               FIELDS 1-N:       Type Specific Real Parameters
 
 
          For Analysis Type = 0, Unknown
 
          RECORD 7:
 
               FIELD 1:   1
               FIELD 2:   1
               FIELD 3:   ID Number
 
          RECORD 8:
 
               FIELD 1:   0.0
 
          For Analysis Type = 1, Static
 
          RECORD 7:
               FIELD 1:    1
               FIELD 2:    1
               FIELD 3:    Load Case Number
 
          RECORD 8:
               FIELD 11:    0.0
 
          For Analysis Type = 2, Normal Mode
 
          RECORD 7:
 
               FIELD 1:    2
               FIELD 2:    4
               FIELD 3:    Load Case Number
               FIELD 4:    Mode Number
 
          RECORD 8:
               FIELD 1:    Frequency (Hertz)
               FIELD 2:    Modal Mass
               FIELD 3:    Modal Viscous Damping Ratio
               FIELD 4:    Modal Hysteretic Damping Ratio
 
          For Analysis Type = 3, Complex Eigenvalue
 
          RECORD 7:
               FIELD 1:    2
               FIELD 2:    6
               FIELD 3:    Load Case Number
               FIELD 4:    Mode Number
 
          RECORD 8:
 
               FIELD 1:    Real Part Eigenvalue
               FIELD 2:    Imaginary Part Eigenvalue
               FIELD 3:    Real Part Of Modal A
               FIELD 4:    Imaginary Part Of Modal A
               FIELD 5:    Real Part Of Modal B
               FIELD 6:    Imaginary Part Of Modal B
 
 
          For Analysis Type = 4, Transient
 
          RECORD 7:
 
               FIELD 1:    2
               FIELD 2:    1
               FIELD 3:    Load Case Number
               FIELD 4:    Time Step Number
 
          RECORD 8:
               FIELD 1: Time (Seconds)
 
          For Analysis Type = 5, Frequency Response
 
          RECORD 7:
 
               FIELD 1:    2
               FIELD 2:    1
               FIELD 3:    Load Case Number
               FIELD 4:    Frequency Step Number
 
          RECORD 8:
               FIELD 1:    Frequency (Hertz)
 
          For Analysis Type = 6, Buckling
 
          RECORD 7:
 
               FIELD 1:    1
               FIELD 2:    1
               FIELD 3:    Load Case Number
 
          RECORD 8:
 
               FIELD 1: Eigenvalue
 
          RECORD 9:      Format (I10)
 
               FIELD 1:          Node Number
 
          RECORD 10:     Format (6E13.5)
               FIELDS 1-N:       Data At This Node (NDV Real Or
                         Complex Values)
 
          Records 9 And 10 Are Repeated For Each Node.
 
          Notes:
          1        Id Lines May Not Be Blank.  If No Information Is
                      Required, The Word "None" Must Appear  Columns 1-4.
 
          2        For Complex Data There Will Be 2*Ndv Data Items At Each
                      Node. The Order Is Real Part For Value 1,  Imaginary
                      Part For Value 1, Etc.
          3        The Order Of Values For Various Data  Characteristics
                      Is:
                          3 DOF Global Vector:
                                  X, Y, Z

                          6 DOF Global Vector:
                                  X, Y, Z,
                                  Rx, Ry, Rz

                          Symmetric Global Tensor:
                                  Sxx, Sxy, Syy,
                                  Sxz, Syz, Szz

                          General Global Tensor:
                                  Sxx, Syx, Szx,
                                  Sxy, Syy, Szy,
                                  Sxz, Syz, Szz

                          Shell And Plate Element Load:
                                  Fx, Fy, Fxy,
                                  Mx, My, Mxy,
                                  Vx, Vy
 
          4        Id Line 1 Always Appears On Plots In Output Display.
          5        If Specific Data Type Is "Unknown," ID Line 2 Is
                      Displayed As Data Type In Output Display.
          6        Typical Fortran I/O Statements For The Data Sections
                      Are:
 
                                   Read(Lun,1000)Num
                                   Write
                          1000 Format (I10)
                                   Read(Lun,1010) (VAL(I),I=1,NDV)
                                   Write
                          1010 format (6e13.5)
 
 
                          Where:     Num Is Node Number 
                                     Val Is Real Or Complex Data  Array
                                     Ndv Is Number Of Data Values  Per Node
 
          7        Data Characteristic Values Imply The Following Values
                      Of Ndv:
                                      Scalar: 1
                                      3 DOF Global Vector: 3
                                      6 DOF Global Vector: 6
                                      Symmetric Global Tensor: 6
                                      General Global Tensor: 9
 
          8        Data Associated With I-DEAS Test Has The Following
                      Special Forms of Specific Data Type and ID Line 5.

                   For Record 6 Field 4-Specific Data Type, values 0
                      through 12 are as defined above.  13 and 15 
                      through 19 are:

                               13: excitation force
                               15: pressure
                               16: mass
                               17: time
                               18: frequency
                               19: rpm

                   The form of ID Line 5 is:
 
                   Format (4I10)
                   FIELD 1:  Reference Coordinate Label

                   FIELD 2:  Reference Coordinate Direction
                                1: X Direction
                               -1: -X Direction
                                2: Y Direction
                               -2: -Y Direction
                                3: Z Direction
                               -3: -Z Direction
 
                   FIELD 3:  Numerator Signal Code
                                see Specific Data Type above
 
                   FIELD 4:  Denominator Signal Code
                                see Specific Data Type above


                   Also note that the modal mass in record 8 is calculated
                   from the parameter table by I-DEAS Test.
 
          9        Any Record With All 0.0's Data Entries Need Not (But
                      May) Appear.
 
          10       A Direct Result Of 9 Is That If No Records 9 And 10
                      Appear, All Data For The Data Set Is 0.0.
 
          11       When New Analysis Types Are Added, Record 7 Fields 1
                      And 2 Are Always > Or = 1 With Dummy Integer And
                      Real Zero Data If Data Is Not Required. If Complex
                      Data Is Needed, It Is Treated As Two Real Numbers,
                      Real Part Followed By Imaginary Point.
 
          12       Dataloaders Use The Following ID Line Convention:
 
                              1.   (80A1) Model
                                  Identification
                              2.   (80A1) Run
                                  Identification
                              3.   (80A1) Run
                                  Date/Time
                              4.   (80A1) Load Case
                                  Name
 
                          For Static:
 
                              5.   (17h Load Case
                                  Number;, I10) For
                                  Normal Mode:
                              5.   (10h Mode Same,
                                  I10, 10H Frequency,
                                  E13.5)
          13       No Maximum Value For Ndv .
 
          14       Typical Fortran I/O Statements For Processing Records 7
                      And 8.

                            Read (LUN,1000)NINT,NRVAL,(IPAR(I),I=1,NINT
                       1000 Format (8I10)
                            Read (Lun,1010) (RPAV(I),I=1,NRVAL)
                       1010 Format (6E13.5)
 
          15       For Situations With Reduced # Dof's, Use 3 DOF
                      Translations Or 6 DOF Translation And Rotation With
                      Unused Values = 0.
 
----------------------------------------------------------------------
"""

    if raw:
        return out
    else:
        print(out)   

def _write55(fh, dset):
    """
    Writes data at nodes - data-set 55 - to an open file fh. Currently:
       - only normal mode (2)
       - complex eigenvalue first order (displacement) (3)
       - frequency response and (5)
       - complex eigenvalue second order (velocity) (7) analyses are supported.
    """
    try:
        # Handle general optional fields
        dset = _opt_fields(dset,
                                {'units_description': ' ',
                                    'id1': 'NONE',
                                    'id2': 'NONE',
                                    'id3': 'NONE',
                                    'id4': 'NONE',
                                    'id5': 'NONE',
                                    'model_type': 1})
        # ... and some data-type specific optional fields
        if dset['analysis_type'] == 2:
            # normal modes
            dset = _opt_fields(dset,
                                    {'modal_m': 0,
                                        'modal_damp_vis': 0,
                                        'modal_damp_his': 0})
        elif dset['analysis_type'] in (3, 7):
            # complex modes
            dset = _opt_fields(dset,
                                    {'modal_b': 0.0 + 0.0j,
                                        'modal_a': 0.0 + 0.0j})
            if not np.iscomplexobj(dset['modal_a']):
                dset['modal_a'] = dset['modal_a'] + 0.j
            if not np.iscomplexobj(dset['modal_b']):
                dset['modal_b'] = dset['modal_b'] + 0.j
        elif dset['analysis_type'] == 5:
            # frequency response
            pass
        else:
            # unsupported analysis type
            raise ValueError('Error writing data-set #55: unsupported analysis type')
        # Response arrays r1, r2, ... give the number of data values per node
        responses = []
        while dset.get('r%i' % (len(responses) + 1)) is not None:
            responses.append(np.asarray(dset['r%i' % (len(responses) + 1)]))
        if not responses:
            raise KeyError('r1')
        n_data_per_node = len(responses)
        complex_data = any(np.iscomplexobj(r) for r in responses)
        # Data type 4 or 6 selects double precision
        double = dset.get('data_type') in (4, 6)
        if complex_data:
            data_type = 6 if double else 5
        else:
            data_type = 4 if double else 2
        # Write strings to the file
        fh.write('%6i\n%6i%74s\n' % (-1, 55, ' '))
        fh.write('%-80s\n' % _fit_string(dset['id1'], 80, 'id1', 55))
        fh.write('%-80s\n' % _fit_string(dset['id2'], 80, 'id2', 55))
        fh.write('%-80s\n' % _fit_string(dset['id3'], 80, 'id3', 55))
        fh.write('%-80s\n' % _fit_string(dset['id4'], 80, 'id4', 55))
        fh.write('%-80s\n' % _fit_string(dset['id5'], 80, 'id5', 55))
        fh.write('%10i%10i%10i%10i%10i%10i\n' %
                    (dset['model_type'], dset['analysis_type'], dset['data_ch'],
                    dset['spec_data_type'], data_type, n_data_per_node))
        if dset['analysis_type'] == 2:
            # Normal modes
            fh.write('%10i%10i%10i%10i\n' % (2, 4, dset['load_case'], dset['mode_n']))
            fh.write('%13.5e%13.5e%13.5e%13.5e\n' % (dset['freq'], dset['modal_m'],
                                                        dset['modal_damp_vis'], dset['modal_damp_his']))
        elif dset['analysis_type'] == 5:
            # Frequenc response
            fh.write('%10i%10i%10i%10i\n' % (2, 1, dset['load_case'], dset['freq_step_n']))
            fh.write('%13.5e\n' % dset['freq'])
        elif (dset['analysis_type'] == 3) or (dset['analysis_type'] == 7):
            # Complex modes
            fh.write('%10i%10i%10i%10i\n' % (2, 6, dset['load_case'], dset['mode_n']))
            fh.write('%13.5e%13.5e%13.5e%13.5e%13.5e%13.5e\n' % (
                dset['eig'].real, dset['eig'].imag, dset['modal_a'].real, dset['modal_a'].imag,
                dset['modal_b'].real, dset['modal_b'].imag))
        else:
            raise ValueError('Unsupported analysis type')
        # Records 9 and 10 of every node: node number, then its values
        # (for complex data the real and imaginary part of each value)
        if complex_data:
            columns = [part for r in responses for part in (r.real, r.imag)]
        else:
            columns = responses
        field, per_line = ('%20.11e', 4) if double else ('%13.5e', 6)
        n_values = len(columns)
        value_lines = [field * min(per_line, n_values - i) for i in range(0, n_values, per_line)]
        fmt = '%10i\n' + '\n'.join(value_lines) + '\n'
        fh.write(''.join(fmt % row for row in zip(dset['node_nums'], *columns)))
        fh.write('%6i\n' % -1)
    except KeyError as msg:
        raise Exception('The required key \'' + msg.args[0] + '\' not present when writing data-set #55')
    except Exception as msg:
        raise Exception('Error writing data-set #55') from msg


def _extract55(block_data):
    """
    Extract data at nodes - data-set 55. Currently:
       - only normal mode (2)
       - complex eigenvalue first order (displacement) (3)
       - frequency response and (5)
       - complex eigenvalue second order (velocity) (7) analyses are supported.
    """
    dset = {'type': 55}
    try:
        split_data = block_data.splitlines(True)
        dset.update(_parse_header_line(split_data[2], 1, [80], [1], ['id1']))
        dset.update(_parse_header_line(split_data[3], 1, [80], [1], ['id2']))
        dset.update(_parse_header_line(split_data[4], 1, [80], [1], ['id3']))
        dset.update(_parse_header_line(split_data[5], 1, [80], [1], ['id4']))
        dset.update(_parse_header_line(split_data[6], 1, [80], [1], ['id5']))
        dset.update(_parse_header_line(split_data[7], 6, [10, 10, 10, 10, 10, 10], [2, 2, 2, 2, 2, 2],
                                            ['model_type', 'analysis_type', 'data_ch', 'spec_data_type',
                                                'data_type', 'n_data_per_node']))
        if dset['analysis_type'] == 2:
            # normal mode
            dset.update(_parse_header_line(split_data[8], 4, [10, 10, 10, 10, 10, 10, 10, 10],
                                                [-1, -1, 2, 2, -1, -1, -1, -1],
                                                ['', '', 'load_case', 'mode_n', '', '', '', '']))
            dset.update(_parse_header_line(split_data[9], 4, [13, 13, 13, 13, 13, 13], [3, 3, 3, 3, -1, -1],
                                                ['freq', 'modal_m', 'modal_damp_vis', 'modal_damp_his', '', '']))
        elif (dset['analysis_type'] == 3) or (dset['analysis_type'] == 7):
            # complex eigenvalue
            dset.update(_parse_header_line(split_data[8], 4, [10, 10, 10, 10, 10, 10, 10, 10],
                                                [-1, -1, 2, 2, -1, -1, -1, -1],
                                                ['', '', 'load_case', 'mode_n', '', '', '', '']))
            dset.update(_parse_header_line(split_data[9], 4, [13, 13, 13, 13, 13, 13], [3, 3, 3, 3, 3, 3],
                                                ['eig_r', 'eig_i', 'modal_a_r', 'modal_a_i', 'modal_b_r',
                                                    'modal_b_i']))
            dset.update({'modal_a': dset['modal_a_r'] + 1.j * dset['modal_a_i']})
            dset.update({'modal_b': dset['modal_b_r'] + 1.j * dset['modal_b_i']})
            dset.update({'eig': dset['eig_r'] + 1.j * dset['eig_i']})
            del dset['modal_a_r'], dset['modal_a_i'], dset['modal_b_r'], dset['modal_b_i']
            del dset['eig_r'], dset['eig_i']
        elif dset['analysis_type'] == 5:
            # frequency response
            dset.update(_parse_header_line(split_data[8], 4, [10, 10, 10, 10, 10, 10, 10, 10],
                                                [-1, -1, 2, 2, -1, -1, -1, -1],
                                                ['', '', 'load_case', 'freq_step_n', '', '', '', '']))
            dset.update(_parse_header_line(split_data[9], 1, [13, 13, 13, 13, 13, 13], [3, -1, -1, -1, -1, -1],
                                                ['freq', '', '', '', '', '']))
        # Records 9 and 10 of every node: node number, then its values
        if dset['data_type'] not in (2, 4, 5, 6):
            raise ValueError('data type %i is not supported' % dset['data_type'])
        complex_data = dset['data_type'] in (5, 6)
        double = dset['data_type'] in (4, 6)
        lines = ''.join(split_data[10:]).splitlines()
        n_data_per_node = dset['n_data_per_node']
        try:
            dset['node_nums'], values = _read_node_values(lines, n_data_per_node * (2 if complex_data else 1), double)
        except ValueError:
            # Some writers give a wrong number of values per node in record 6;
            # the data characteristic implies the number (note 7 of the specification)
            n_implied = {1: 1, 2: 3, 3: 6, 4: 6, 5: 9}.get(dset['data_ch'])
            if n_implied in (None, n_data_per_node):
                raise
            dset['node_nums'], values = _read_node_values(lines, n_implied * (2 if complex_data else 1), double)
            warnings.warn('Data-set 55: record 6 gives %i values per node, but the data has %i values per node, '
                          'as implied by the data characteristic %i; %i values are read.'
                          % (n_data_per_node, n_implied, dset['data_ch'], n_implied), stacklevel=4)
            n_data_per_node = dset['n_data_per_node'] = n_implied
        for i in range(n_data_per_node):
            if complex_data:
                dset['r%i' % (i + 1)] = values[:, 2 * i] + 1.j * values[:, 2 * i + 1]
            else:
                dset['r%i' % (i + 1)] = values[:, i].copy()
    except Exception as msg:
        raise Exception('Error reading data-set #55') from msg
    return dset


def _split_values(lines, double):
    """Returns the values in record 10 lines as strings.

    Single precision values are in fields of 13 characters, which may touch
    each other; spaces that fill a line up to 80 characters are ignored.
    Double precision values are separated by spaces and may use the Fortran
    exponent D.
    """
    if double:
        return ' '.join(lines).replace('D', 'E').replace('d', 'e').split()
    return [line[i:i + 13] for line in lines for i in range(0, len(line.rstrip()) - 12, 13)]


def _read_node_values(lines, n_values, double):
    """Returns the node numbers and an array with n_values values of each node.

    Every node has a line with its number (record 9), followed by as many
    lines of values (record 10) as are needed for n_values values.
    """
    if n_values < 1:
        raise ValueError('number of data values per node must be at least 1')
    # Lines of one node: its number and the value lines that follow it
    lines_per_node = 1
    n_found = 0
    while n_found < n_values and lines_per_node < len(lines):
        n_found += len(_split_values([lines[lines_per_node]], double))
        lines_per_node += 1
    node_nums = np.array([int(line) for line in lines[::lines_per_node]], dtype=int)
    value_lines = lines.copy()
    del value_lines[::lines_per_node]
    values = np.array(_split_values(value_lines, double), 'd')
    if len(lines) % lines_per_node or values.size != len(node_nums) * n_values:
        raise ValueError('expected %i values at each of the %i nodes, found %i values in total'
                         % (n_values, len(node_nums), values.size))
    return node_nums, values.reshape(len(node_nums), n_values)


def prepare_55(
        id1=None,
        id2=None,
        id3=None,
        id4=None,
        id5=None,
        model_type=None,
        analysis_type=None,
        data_ch=None,
        spec_data_type=None,
        data_type=None,
        n_data_per_node=None,
        r1=None,
        r2=None,
        r3=None,
        r4=None,
        r5=None,
        r6=None,
        load_case=None,
        mode_n=None,
        freq=None,
        modal_m=None,
        modal_damp_vis=None,
        modal_damp_his=None,
        eig=None,
        modal_a=None,
        modal_b=None,
        freq_step_n=None,
        node_nums=None,
        return_full_dict=False):
    """Prepare data-set 55 (data at nodes) for writing.

    Data-set 55 holds one set of values at the nodes of a model: one or more real or
    complex values per node, e.g. a mode shape (normal or complex mode) or the response
    shape at one frequency of a frequency response analysis. Each mode or frequency is a
    separate data-set. The node coordinates are given in data-set 15 or 2411.

    pyuff writes the analysis types 2 (normal mode), 3 and 7 (complex eigenvalue) and
    5 (frequency response). The fields of records 7 and 8 depend on the analysis type;
    the fields not used by the given analysis type are ignored.

    Parameters
    ----------
    id1 : str, optional
        ID line 1, e.g. the model identification, up to 80 characters (record 1,
        field 1). Default: ``'NONE'``.
    id2 : str, optional
        ID line 2, e.g. the run identification, up to 80 characters (record 2,
        field 1). Default: ``'NONE'``.
    id3 : str, optional
        ID line 3, e.g. the run date and time, up to 80 characters (record 3,
        field 1). Default: ``'NONE'``.
    id4 : str, optional
        ID line 4, e.g. the load case name, up to 80 characters (record 4,
        field 1). Default: ``'NONE'``.
    id5 : str, optional
        ID line 5, e.g. the mode number and frequency, up to 80 characters (record 5,
        field 1). Default: ``'NONE'``.
    model_type : {0, 1, 2, 3}, optional
        Model type (record 6, field 1): 0 unknown, 1 structural, 2 heat transfer,
        3 fluid flow. Default: 1.
    analysis_type : {2, 3, 5, 7}
        Analysis type (record 6, field 2): 2 normal mode, 3 complex eigenvalue first
        order, 5 frequency response, 7 complex eigenvalue second order. The
        specification also defines 0 unknown, 1 static, 4 transient and 6 buckling,
        which pyuff does not write.
    data_ch : {0, 1, 2, 3, 4, 5}
        Data characteristic (record 6, field 3), which implies the number of values
        per node: 0 unknown, 1 scalar (1 value), 2 3-DOF global translation vector
        (3 values: X, Y, Z), 3 6-DOF global translation and rotation vector
        (6 values: X, Y, Z, RX, RY, RZ), 4 symmetric global tensor (6 values: SXX,
        SXY, SYY, SXZ, SYZ, SZZ), 5 general global tensor (9 values). pyuff does not
        check that it agrees with the number of response arrays ``r1``, ``r2``, ...
    spec_data_type : int
        Specific data type, 0 to 18 (record 6, field 4): 0 unknown, 1 general,
        2 stress, 3 strain, 4 element force, 5 temperature, 6 heat flux, 7 strain
        energy, 8 displacement, 9 reaction force, 10 kinetic energy, 11 velocity,
        12 acceleration, 13 strain energy density, 14 kinetic energy density,
        15 hydro-static pressure, 16 heat gradient, 17 code checking value,
        18 coefficient of pressure. For data from I-DEAS Test, 13 is excitation force,
        15 pressure, 16 mass, 17 time and 18 frequency.
    data_type : {2, 4, 5, 6}, optional
        Data type (record 6, field 5): 2 real single precision, 4 real double
        precision, 5 complex single precision, 6 complex double precision. Whether
        the data is real or complex is set by pyuff from the response arrays (complex
        if any of them is complex); this parameter only selects the precision: 4 or 6
        writes the values in double precision, any other value or none in single
        precision (6E13.5). Double precision is not part of the data-set 55
        specification, which knows only data types 2 and 5; the values are written as
        4E20.12, as in data-set 58, and some readers (e.g. SDynPy) do not read them. A
        data-set read from a double precision file keeps data type 4 or 6 and is
        therefore written in double precision. Default: single precision.
    n_data_per_node : int, optional
        Number of data values per node (record 6, field 6). Ignored when writing:
        pyuff writes the number of response arrays given (``r1``, ``r2``, ... up to
        the first one that is missing or None).
    r1, r2, r3, r4, r5, r6 : array_like of float or complex
        Response values at the nodes, one array per value at a node, each with the
        same length as ``node_nums`` (record 10). The order follows ``data_ch``: for a
        scalar give ``r1``; for a 3-DOF translation ``r1``, ``r2``, ``r3`` are the X,
        Y and Z components; for a 6-DOF vector ``r4``, ``r5``, ``r6`` are the
        rotations RX, RY, RZ; for a symmetric tensor ``r1`` to ``r6`` are SXX, SXY,
        SYY, SXZ, SYZ, SZZ. ``r1`` is required; the number of arrays given sets the
        number of values per node. For a general tensor (9 values) add the keys
        ``'r7'``, ``'r8'`` and ``'r9'`` to the returned dictionary. If any array is
        complex, all values are written as complex (real and imaginary part).
    load_case : int
        Load case number (record 7, field 3). Required for all analysis types.
    mode_n : int
        Mode number (record 7, field 4). Required for analysis types 2, 3 and 7.
    freq : float
        Frequency in Hz (record 8, field 1). Required for analysis types 2 (natural
        frequency) and 5 (frequency of the response).
    modal_m : float, optional
        Modal mass (record 8, field 2). Used for analysis type 2. Default: 0.
    modal_damp_vis : float, optional
        Modal viscous damping ratio (record 8, field 3). Used for analysis type 2.
        Default: 0.
    modal_damp_his : float, optional
        Modal hysteretic damping ratio (record 8, field 4). Used for analysis
        type 2. Default: 0.
    eig : complex
        Eigenvalue; its real and imaginary part are written (record 8, fields 1 and
        2). Required for analysis types 3 and 7.
    modal_a : complex, optional
        Modal A; its real and imaginary part are written (record 8, fields 3 and 4).
        Used for analysis types 3 and 7. Default: 0.
    modal_b : complex, optional
        Modal B; its real and imaginary part are written (record 8, fields 5 and 6).
        Used for analysis types 3 and 7. Default: 0.
    freq_step_n : int
        Frequency step number (record 7, field 4). Required for analysis type 5.
    node_nums : array_like of int
        Node numbers, one for each value in ``r1``, ``r2``, ... (record 9, field 1).
    return_full_dict : bool, optional
        If True, the dictionary has all keys of the data-set, with ``None`` for
        the values not given; if False (default), only the given values.

    Returns
    -------
    dict
        The data-set, to be written with :meth:`pyuff.UFF.write_sets`.

    Examples
    --------
    Write the first two mode shapes of a model with four nodes, as 3-DOF translations:

    >>> import numpy as np
    >>> import pyuff
    >>> node_nums = np.array([1, 2, 3, 4])
    >>> freqs = [10.0, 25.0]
    >>> phi = np.random.default_rng(0).normal(size=(2, 3, len(node_nums)))
    >>> mode_shapes = [
    ...     pyuff.prepare_55(
    ...         id1='Beam mode shapes',
    ...         analysis_type=2,
    ...         data_ch=2,
    ...         spec_data_type=8,
    ...         load_case=1,
    ...         mode_n=i + 1,
    ...         freq=freqs[i],
    ...         modal_damp_vis=0.01,
    ...         node_nums=node_nums,
    ...         r1=phi[i, 0],
    ...         r2=phi[i, 1],
    ...         r3=phi[i, 2])
    ...     for i in range(len(freqs))]
    >>> pyuff.UFF('mode_shapes.uff').write_sets(mode_shapes, mode='overwrite')
    """

    if type(id1) != str and id1 != None:
        raise TypeError('id1 must be string.')
    if type(id2) != str and id2 != None:
        raise TypeError('id2 must be string.')
    if type(id3) != str and id3 != None:
        raise TypeError('id3 must be string.')
    if type(id4) != str and id4 != None:
        raise TypeError('id4 must be string.')
    if type(id5) != str and id5 != None:
        raise TypeError('id5 must be string.')
    if model_type not in (0, 1, 2, 3, None):
        raise ValueError('model_type can be 0:Unknown, 1:Structural, 2:Heat Transfer, 3:Fluid Flow')
    if analysis_type not in (2, 3, 5, 7, None):
        raise ValueError('analysis_type: Currently only only normal mode (2), complex eigenvalue first order (displacement) (3), frequency response and (5) and complex eigenvalue second order (velocity) (7) are supported.')
    if data_ch not in (0, 1, 2, 3, 4, 5, None):
        raise ValueError('data_ch can be: 0,1,2,4,5')
    if spec_data_type not in np.arange(19) and spec_data_type != None:
        raise ValueError('spec_data_type must be integer between 0 and 18')
    if data_type not in (2, 4, 5, 6, None):
        raise ValueError('data_type can be 2:Real, 4:Real double precision, 5:Complex or 6:Complex double precision')
    if np.array(n_data_per_node).dtype != int and n_data_per_node != None:
        raise TypeError('n_data_per_node must be integers')
    if not np.issubdtype(np.array(r1).dtype, np.number) and r1 != None:
        raise TypeError('r1 must have numeric (float or complex) values')
    if not np.issubdtype(np.array(r2).dtype, np.number) and r2 != None:
        raise TypeError('r2 must have numeric (float or complex) values')
    if not np.issubdtype(np.array(r3).dtype, np.number) and r3 != None:
        raise TypeError('r3 must have numeric (float or complex) values')
    if not np.issubdtype(np.array(r4).dtype, np.number) and r4 != None:
        raise TypeError('r4 must have numeric (float or complex) values')
    if not np.issubdtype(np.array(r5).dtype, np.number) and r5 != None:
        raise TypeError('r5 must have numeric (float or complex) values')
    if not np.issubdtype(np.array(r6).dtype, np.number) and r6 != None:
        raise TypeError('r6 must have numeric (float or complex) values')
    if type(load_case) != int and load_case != None:
        raise TypeError('load_case must be integer')
    if type(mode_n) != int and mode_n != None:
        raise TypeError('mode_n must be integer')
    if np.array(mode_n).dtype != int and mode_n != None:
        raise TypeError('r6 must be integer')
    if np.array(freq).dtype != float and freq != None:
        raise TypeError('freq must be float')
    if type(modal_damp_vis) != float and modal_damp_vis != None:
        raise TypeError('modal_damp_vis must be float')
    if type(modal_damp_his) != float and modal_damp_his != None:
        raise TypeError('modal_damp_his must be float')
    if not np.issubdtype(np.array(eig).dtype, np.number) and eig != None:
        raise TypeError('eig must be numeric (float or complex)')
    if not np.issubdtype(np.array(modal_a).dtype, np.number) and modal_a != None:
        raise TypeError('modal_a must be numeric (float or complex)')
    if not np.issubdtype(np.array(modal_b).dtype, np.number) and modal_b != None:
        raise TypeError('modal_b must be numeric (float or complex)')
    if type(freq_step_n) != int and freq_step_n != None:
        raise TypeError('freq_step_n must be int')
    if np.array(node_nums).dtype != int and node_nums != None:
        raise TypeError('mode_nums must be int')

    dataset = {
        'type': 55,
        'id1': id1,
        'id2': id2,
        'id3': id3,
        'id4': id4,
        'id5': id5,
        'model_type':model_type,
        'analysis_type': analysis_type,
        'data_ch': data_ch,
        'spec_data_type': spec_data_type,
        'data_type': data_type,
        'n_data_per_node': n_data_per_node,
        'r1': r1,
        'r2': r2,
        'r3': r3,
        'r4': r4,
        'r5': r5,
        'r6': r6,
        'load_case': load_case,
        'mode_n': mode_n,
        'freq': freq,
        'modal_m': modal_m,
        'modal_damp_vis': modal_damp_vis,
        'modal_damp_his': modal_damp_his,
        'eig': eig,
        'modal_a': modal_a,
        'modal_b': modal_b,
        'freq_step_n': freq_step_n,
        'node_nums': node_nums
        }

    if return_full_dict is False:
        dataset = check_dict_for_none(dataset)

    return dataset


