Supported datasets
==================

Dataset:15
-------------

Type number ``'type'``: 15

`File structure <https://www.ceas3.uc.edu/sdrluff/view.php>`_

* Record 1:
    * Field 1: node label ``'node_nums'``
    * Field 2: deformation coordinate system numbers ``<'def_cs'>``
    * Field 3: displacement coordinate system numbers <'disp_cs'>
    * Field 4: color ``<'color'>``
    * Field 5-7: 3 - Dimensional coordinates of node in the definition system ``'x'`` ``'y'`` ``'z'``




Dataset: 55
-------------

Type number ``'type'``: 55

* Record 1:
    *Field 1: ID Line 1 ``<'id1'>``
* Record 2:
    *Field 2: ID Line 2 ``<'id2'>``
* Record 3:
    *Field 3: ID Line 3 ``<'id3'>``
* Record 4:
    *Field 4: ID Line 4 ``<'id4'>``
* Record 5:
    *Field 5: ID Line 5 ``<'id5'>``
* Record 6:
    * Field 1: Model type ``<'model_type'>``
    * Field 2: Analysis type ``'analysis_type'`` currently only only normal mode (2), complex eigenvalue first order (displacement) (3), frequency response and (5) and complex eigenvalue second order (velocity) (7) are supported
    * Field 3: Data characteristic number ``'data_ch'``
    * Field 4: Specific data type ``'spec_data_type'``
    * Field 5: Data type ``<<'data_type'>>``
    * Field 6: Number of data values per node ``<<'n_data_per_node'>>``

Records 7 And 8 Are Analysis Type Specific

General Form:

* Record 7:
    * Field 1: Number of integer data values
    * Field 2: Number of real data values
    * Fields 3-N: Type specific integer parameters
* Record 8:
    * Fields 1-N: Type specific real parameters

**For Analysis Type = 0, Unknown:**

* Record 7:
    * Field 1: 1
    * Field 2: 1
    * Field 3: ID number
* Record 8:
    * Field 1: 0.0

**For Analysis Type = 1, Static:**

* Record 7:
    * Field 1: 1
    * Field 2: 1
    * Field 3: Load case number ``'load_case'``
* Record 8:
    * Field 1: 0.0

**For Analysis Type = 2, Normal Mode:**

* Record 7:
    * Field 1: 2
    * Field 2: 4
    * Field 3: Load case number ``'load_case'``
    * Filed 4: Mode number ``'mode_n'``
* Record 8:
    * Field 1: Frequency (Hertz)  ``'freq'``
    * Field 2: Modal mass  ``<'modal_m'>``
    * Field 3: MOdal viscous damping ratio  ``<'modal_damp_vis'>``
    * Filed 4: Modal hysteric damping ratio  ``<'modal_damp_his'>``

**For Analysis Type = 3, Complex Eigenvalue:**

 * Record 7:
    * Field 1: 2
    * Field 2: 6
    * Field 3: Load case number ``'load_case'``
    * Filed 4: Mode number ``'mode_n'``
* Record 8:
    * Field 1: Real Part Eigenvalue ``'eig'``
    * Field 2: Imaginary Part Eigenvalue ``'eig'``
    * Field 3: Real Part Of Modal A ``<'modal_a'>``
    * Filed 4: Imaginary Part Of Modal A ``<'modal_a'>``
    * Field 5: Real Part Of Modal B ``<'modal_b'>``
    * Field 6: Imaginary Part Of Modal B ``<'modal_b'>``

**For Analysis Type = 4, Transient**

* Record 7:
    * Field 1: 2
    * Field 2: 1
    * Field 3: Load case number ``'load_case'``
    * Filed 4: Time step number
* Record 8:
    * Field 1: Time(seconds)

 **For Analysis Type = 5, Frequency Response**

 * Record 7:
    * Field 1: 2
    * Field 2: 1
    * Field 3: Load case number ``'load_case'``
    * Filed 4: Frequency step number ``'freq_step_n'``
* Record 8:
    * Field 1: Frequency(Hertz) ``'freq'``




* Record 9:
    * Field 1: Node number ``'node_nums'``
* Record 10:
    * Fields 1-N: Data at this node

Records 9 And 10 Are Repeated For Each Node.

Dataset: 58
-------------
Type number ``'type'``: 58

* Record 1:
    * Field 1: ID Line 1 (Generally used for function description)
* Record 2:
    * Field 1: ID Line 2
* Record 3:
    * Field 1: ID Line 3 (Generally used for date and time DD-MMM-YY and HH:MM:SS, with general format(9A1,1X,8A1))
* Record 4:
    * Field 1: ID Line 4
* Record 5:
    * Field 1: ID Line 5
* Record 6: DOF Identification
    * Field 1: Function type ``'func_type'`` ; only 1, 2, 3, 4 and 6 are supported
    * Field 2: Function identification number
    * Field 3: Version number, or sequence number ``<'ver_num'>``
    * Field 4: Load case identification number ``<'load_case_id'>`` 0 - Single Point Excitation
    * Field 5: Response entity name ("None" if unused) ``<'rsp_ent_name'>``
    * Field 6: Response node ``'rsp_node'``
    * Field 7: Response direction ``'rsp_dir'``
    * Field 8: Reference entity name ("None" if unused) ``<'ref_ent_name'>``
    * Field 9: Reference node ``'ref_node'``
    * Field 10: Reference direction ``'ref_dir'``

Fields 8, 9, and 10 are only relevant if field 4 is zero.

* Record 7: Data Form
    * Field 1: Ordinate Data Type ``<<'ord_data_type'>>``
    * Field 2: Number of data pairs for uneven abscissa spacing, or number of data values for even abscissa spacing ``<<'num_pts'>>``
    * Field 3: Abscissa spacing ( 0=uneven, 1=even)``<<'abscissa_spacing'>>`` 
    * Field 4: Abscissa minimum (0.0 if spacing uneven) ``<<'abscissa_min'>>`` 
    * Field 5: Abscissa increment (0.0 if spacing uneven) ``<<'abscissa_inc'>>``
    * Field 6: Z-axis value (0.0 if unused) ``<'z_axis_value'>``

* Record 8: Abscissa Data Characteristics
    * Field 1: Specific data type ``<'spec_data_type'>``
    * Field 2: Length units exponent ``<'abscissa_len_unit_exp'>``
    * Field 3: Force units exponent ``<'abscissa_force_unit_exp'>``
    * Field 4: Temperature units exponent ``<'abscissa_temp_unit_exp'>``
    * Field 5: Axis label ("NONE" if not used) ````
    * Field 6: Axis units label ("NONE" if not used) ````



Dataset: 58b
-------------

Dataset: 82
-------------

Type number ``'type'``: 82

Dataset: 151
-------------

Type number ``'type'``: 58

Dataset: 164
-------------

Type number ``'type'``: 58

Dataset: 2411
-------------

Type number ``'type'``: 58

Dataset: 2412
-------------

Type number ``'type'``: 58

Dataset: 2420
-------------

Type number ``'type'``: 58