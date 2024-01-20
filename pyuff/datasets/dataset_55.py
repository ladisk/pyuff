import numpy as np

from ..tools import _opt_fields, _parse_header_line, check_dict_for_none

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
        # Some additional checking
        data_type = 2
        #             if dset.has_key('r4') and dset.has_key('r5') and dset.has_key('r6'):
        if ('r4' in dset) and ('r5' in dset) and ('r6' in dset):
            n_data_per_node = 6
        else:
            n_data_per_node = 3
        if np.iscomplexobj(dset['r1']):
            data_type = 5
        else:
            data_type = 2
        # Write strings to the file
        fh.write('%6i\n%6i%74s\n' % (-1, 55, ' '))
        fh.write('%-80s\n' % dset['id1'])
        fh.write('%-80s\n' % dset['id2'])
        fh.write('%-80s\n' % dset['id3'])
        fh.write('%-80s\n' % dset['id4'])
        fh.write('%-80s\n' % dset['id5'])
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
        n = len(dset['node_nums'])
        if data_type == 2:
            # Real data
            if n_data_per_node == 3:
                for k in range(0, n):
                    fh.write('%10i\n' % dset['node_nums'][k])
                    fh.write('%13.5e%13.5e%13.5e\n' % (dset['r1'][k], dset['r2'][k], dset['r3'][k]))
            else:
                for k in range(0, n):
                    fh.write('%10i\n' % dset['node_nums'][k])
                    fh.write('%13.5e%13.5e%13.5e%13.5e%13.5e%13.5e\n' %
                                (dset['r1'][k], dset['r2'][k], dset['r3'][k], dset['r4'][k], dset['r5'][k],
                                dset['r6'][k]))
        elif data_type == 5:
            # Complex data; n_data_per_node is assumed being 3
            for k in range(0, n):
                fh.write('%10i\n' % dset['node_nums'][k])
                fh.write('%13.5e%13.5e%13.5e%13.5e%13.5e%13.5e\n' %
                            (dset['r1'][k].real, dset['r1'][k].imag, dset['r2'][k].real, dset['r2'][k].imag,
                            dset['r3'][k].real, dset['r3'][k].imag))
        else:
            raise ValueError('Unsupported data type')
        fh.write('%6i\n' % -1)
    except KeyError as msg:
        raise Exception('The required key \'' + msg.args[0] + '\' not present when writing data-set #55')
    except:
        raise Exception('Error writing data-set #55')


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
            # Body
        split_data = ''.join(split_data[10:])
        node_nums = np.asarray([int(str) for str in split_data.splitlines(True)[::2]])
        node_vals = np.array([[str[i*13:(i+1)*13] for i in range(len(str)//13) ] for str in split_data.splitlines(True)[1::2]], 'd').flatten()
        if dset['data_type'] == 2:
            # real data
            if dset['n_data_per_node'] == 3:
                dset['node_nums'] = node_nums.copy()
                dset['r1'] = node_vals[0::3].copy()
                dset['r2'] = node_vals[1::3].copy()
                dset['r3'] = node_vals[2::3].copy()
            else:
                dset['node_nums'] = node_nums.copy()
                dset['r1'] = node_vals[0::6].copy()
                dset['r2'] = node_vals[1::6].copy()
                dset['r3'] = node_vals[2::6].copy()
                dset['r4'] = node_vals[3::6].copy()
                dset['r5'] = node_vals[4::6].copy()
                dset['r6'] = node_vals[5::6].copy()
        elif dset['data_type'] == 5:
            # complex data
            if dset['n_data_per_node'] == 3:
                dset['node_nums'] = node_nums.copy()
                dset['r1'] = node_vals[0::6].copy() + 1.j * node_vals[1::6].copy()
                dset['r2'] = node_vals[2::6].copy() + 1.j * node_vals[3::6].copy()
                dset['r3'] = node_vals[4::6].copy() + 1.j * node_vals[5::6].copy()
            else:
                raise Exception('Cannot handle 6 points per node and complex data when reading data-set #55')
        else:
            raise Exception('Error reading data-set #55')
    except:
        raise Exception('Error reading data-set #55')
    del node_nums
    del node_vals
    return dset


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
    """
    Name:   Data at Nodes

    R-Record, F-Field

    :param id1: R1 F1, ID Line 1, optional
    :param id2: R2 F1, ID Line 2, optional
    :param id3: R3 F1, ID Line 3, optional
    :param id4: R4 F1, ID Line 4, optional
    :param id5: R5 F1, ID Line 5, optional

    :param model_type: R6 F1, Model type, optional
    :param analysis_type: R6 F2, Analysis type; currently only only normal mode (2), complex eigenvalue first order (displacement) (3), frequency response and (5) and complex eigenvalue second order (velocity) (7) are supported
    :param data_ch: R6 F3, Data characteristic number
    :param spec_data_type: R6 F4, Specific data type
    :param data_type: R6 F5,  Data type, ignored
    :param n_data_per_node: R6 F6, Number of data values per node, ignored

    R7 and R8 are analysis type specific:

    :param r1: Response array for DOF 1,
    :param r2: Response array for DOF 2,
    :param r3: Response array for DOF 3,
    :param r4: Response array for DOF 4,
    :param r5: Response array for DOF 5,
    :param r6: Response array for DOF 6,
    :param load_case: R7 F3, Load case number 
    :param mode_n: R7 F4, Mode number
    :param freq: R8 F1, Frequency (Hertz) 
    :param modal_m: R8 F2, Modal mass, optional
    :param modal_damp_vis: R8 F3, Modal viscous damping ratio, optional
    :param modal_damp_his: R8 F4, Modal hysteric damping ratio, optional
    :param eig: R8 F1: Real part Eigenvalue, R8 F2: Imaginary part Eigenvalue
    :param modal_a: R8 F3: Real part of Modal A, R8 F4: Imaginary part of Modal A, optional
    :param modal_b: R8 F5: Real part of Modal B, R8 F6: Imaginary part of Modal B, optional
    :param freq_step_n: R7 F4, Frequency step number
    :param node_nums: R9 F1 Node number

    :param return_full_dict: If True full dict with all keys is returned, else only specified arguments are included

    **Test prepare_55**

    >>> save_to_file = 'test_pyuff'
    >>> if save_to_file:
    >>>     if os.path.exists(save_to_file):
    >>>         os.remove(save_to_file)
    >>> uff_datasets = []
    >>> modes = [1, 2, 3]
    >>> node_nums = [1, 2, 3, 4]
    >>> freqs = [10.0, 12.0, 13.0]
    >>> for i, b in enumerate(modes):
    >>>     mode_shape = np.random.normal(size=len(node_nums))
    >>>     name = 'TestCase'
    >>>     data = pyuff.prepare_55(
    >>>         model_type=1,
    >>>         id1='NONE',
    >>>         id2='NONE',
    >>>         id3='NONE',
    >>>         id4='NONE',
    >>>         id5='NONE',
    >>>         analysis_type=2,
    >>>         data_ch=2,
    >>>         spec_data_type=8,
    >>>         data_type=2,
    >>>         r1=mode_shape,
    >>>         r2=mode_shape,
    >>>         r3=mode_shape,
    >>>         n_data_per_node=3,
    >>>         node_nums=[1, 2, 3, 4],
    >>>         load_case=1,
    >>>         mode_n=i + 1,
    >>>         modal_m=0,
    >>>         freq=freqs[i],
    >>>         modal_damp_vis=0.,
    >>>         modal_damp_his=0.)
    >>>     uff_datasets.append(data.copy())
    >>>     if save_to_file:
    >>>         uffwrite = pyuff.UFF(save_to_file)
    >>>         uffwrite._write_set(data, 'add')
    >>> uff_datasets
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
    if data_type not in (2, 5, None):
        raise ValueError('data_type can be 2:Real or 5:Complex')
    if np.array(n_data_per_node).dtype != int and n_data_per_node != None:
        raise TypeError('n_data_per_node must be integers')
    if np.array(r1).dtype != float and r1 != None:
        raise TypeError('r1 must have float values')
    if np.array(r2).dtype != float and r2 != None:
        raise TypeError('r2 must have float values')
    if np.array(r3).dtype != float and r3 != None:
        raise TypeError('r3 must have float values')
    if np.array(r4).dtype != float and r4 != None:
        raise TypeError('r4 must have float values')
    if np.array(r5).dtype != float and r5 != None:
        raise TypeError('r5 must have float values')
    if np.array(r6).dtype != float and r6 != None:
        raise TypeError('r6 must have float values')
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
    if np.array(eig).dtype != float and eig != None:
        raise TypeError('eig must be float')
    if np.array(modal_a).dtype != float and modal_a != None:
        raise TypeError('modal_a must be float')   
    if np.array(modal_b).dtype != float and modal_b != None:
        raise TypeError('modal_b must be float')
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


