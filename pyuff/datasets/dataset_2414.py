import numpy as np
import math as math

from ..tools import _opt_fields, _parse_header_line, check_dict_for_none

def get_structure_2414(raw=False):
    """(source: https://www.ceas3.uc.edu/sdrluff/"""
    out = """
Universal Dataset Number: 2414

Name:   Analysis Data
-----------------------------------------------------------------------
 
Record 1:        FORMAT(1I10)
                 Field 1       -- Analysis dataset label
 
Record 2:        FORMAT(40A2)
                 Field 1       -- Analysis dataset name

Record 3:        FORMAT (1I10)
                 Field 1:      -- Dataset location
                                   1:    Data at nodes
                                   2:    Data on elements
                                   3:    Data at nodes on elements
                                   5:    Data at points

Record 4:        FORMAT (40A2)
                 Field 1:      -- ID line 1

Record 5:        FORMAT (40A2)
                 Field 1:      -- ID line 2

Record 6:        FORMAT (40A2)
                 Field 1:      -- ID line 3

Record 7:        FORMAT (40A2)
                 Field 1:      -- ID line 4

Record 8:        FORMAT (40A2)
                 Field 1:      -- ID line 5

Record 9:        FORMAT (6I10)
                 Field 1:      -- Model type 
                                   0:   Unknown
                                   1:   Structural
                                   2:   Heat transfer
                                   3:   Fluid flow
                 Field 2:      -- Analysis type 
                                   0:   Unknown
                                   1:   Static
                                   2:   Normal mode
                                   3:   Complex eigenvalue first order
                                   4:   Transient
                                   5:   Frequency response
                                   6:   Buckling
                                   7:   Complex eigenvalue second order
                                   9:   Static non-linear
                 Field 3:      -- Data characteristic     
                                   0:   Unknown
                                   1:   Scalar
                                   2:   3 DOF global translation vector
                                   3:   6 DOF global translation & rotation 
                                         vector
                                   4:   Symmetric global tensor
                                   6:   Stress resultants
                 Field 4:      -- Result type
                                   2:   Stress
                                   3:   Strain
                                   4:   Element force
                                   5:   Temperature
                                   6:   Heat flux
                                   7:   Strain energy
                                   8:   Displacement
                                   9:   Reaction force
                                   10:  Kinetic energy
                                   11:  Velocity
                                   12:  Acceleration
                                   13:  Strain energy density
                                   14:  Kinetic energy density
                                   15:  Hydro-static pressure
                                   16:  Heat gradient
                                   17:  Code checking value
                                   18:  Coefficient of pressure
                                   19:  Ply stress
                                   20:  Ply strain
                                   21:  Failure index for ply
                                   22:  Failure index for bonding
                                   23:  Reaction heat flow
                                   24:  Stress error density
                                   25:  Stress variation
                                   27:  Shell and plate elem stress resultant
                                   28:  Length
                                   29:  Area
                                   30:  Volume
                                   31:  Mass
                                   32:  Constraint forces
                                   34:  Plastic strain
                                   35:  Creep strain
                                   36:  Strain energy error
                                   37:  Dynamic stress at nodes
                                   38:  Heat Transfer coefficient
                                   39:  Temperature gradient
                                   40:  Kinetic energy dissipation rate
                                   41:  Strain energy error
                                   42:  Mass flow
                                   43:  Mass flux
                                   44:  Heat flow
                                   45:  View factor
                                   46:  Heat load
                                   47:  Stress Component
                                   93:  Unknown
                                   94:  Unknown scalar
                                   95:  Unknown 3DOF vector
                                   96:  Unknown 6DOF vector
                                   97:  Unknown symmetric tensor
                                   98:  Unknown global tensor
                                   99:  Unknown shell and plate resultant
                                  301:  Sound Pressure
                                  302:  Sound Power
                                  303:  Sound Intensity
                                  304:  Sound Energy
                                  305:  Sound Energy Density
                                >1000:  User defined result type
                 Field 5:      -- Data type               
                                   1:   Integer
                                   2:   Single precision floating point
                                   4:   Double precision floating point
                                   5:   Single precision complex
                                   6:   Double precision complex
                 Field 6:      -- Number of data values for the data 
                                  component (NVALDC)

Record 10:       FORMAT (8I10)
                 Field 1:      -- Integer analysis type specific data (1-8)

Record 11:       FORMAT (8I10)
                 Field 1:      -- Integer analysis type specific data (9,10)

Record 12:       FORMAT (6E13.5)
                 Field 1:      -- Real analysis type specific data (1-6)

Record 13:       FORMAT (6E13.5)
                 Field 1:      -- Real analysis type specific data (7-12)

Note: See chart below for specific analysis type information.

Dataset class: Data at nodes

Record 14:       FORMAT (I10)
                 Field 1:      -- Node number

Record 15:       FORMAT (6E13.5)
                 Fields 1-N:   -- Data at this node (NDVAL real or complex

                                                     values)
          
                 Note: Records 14 and 15 are repeated for each node.

Dataset class: Data at elements

Record 14:       FORMAT (2I10)
                 Field 1:      -- Element number
                 Field 2:      -- Number Of data values For this element(NDVAL)
 
Record 15:       FORMAT (6E13.5)
                 Fields 1-N:   -- Data on element(NDVAL Real Or Complex Values)
 
 
                 Note: Records 14 and 15 are repeated for all elements.

Dataset class: Data at nodes on elements

RECORD 14:       FORMAT (4I10)
                 Field 1:      -- Element number
                 Field 2:      -- Data expansion code (IEXP) 
                                  1: Data present for all nodes
                                  2: Data present for only 1st node -All other
                                     nodes the same.
                 Field 3:      -- Number of nodes on elements (NLOCS)
                 Field 4:      -- Number of data values per node (NVLOC)
 
RECORD 15:       FORMAT (6E13.5)
 
                 Fields 1-N:   -- Data Values At Node 1 (NVLOC Real Or
                                  Complex Values)

                 Note:  Records 14 And 15 Are repeated For each Element.  
   
                
                        For Iexp = 1 Record 15 Is repeated NLOCS Times
 
                        For Iexp = 2 Record 15 appears once

Dataset class: Data at points

RECORD 14:       FORMAT (5I10)
                 Field 1:      -- Element number
                 Field 2:      -- Data expansion code (IEXP) 
                                  1: Data present for all points
                                  2: Data present for only 1st point -All other
                                     points the same.
                 Field 3:      -- Number of points on elements (NLOCS)
                 Field 4:      -- Number of data values per point (NVLOC)
                 Field 5:      -- Element order
 
RECORD 15:       FORMAT (6E13.5)
 
                 Fields 1-N:   -- Data Values At point 1 (NVLOC Real Or
                                  Complex Values)

                 Note:  Records 14 And 15 Are repeated For each Element.  
   
                
                        For Iexp = 1 Record 15 Is repeated NLOC Times
 
                        For Iexp = 2 Record 15 appears once          

          Notes:   1.  ID lines may not be blank.  If no information
                       is required, the word "NONE" must appear in
                       columns 1-4.

                   2.  The data is store in 
                       "node-layer-data charateristic" format.

                        Loc1 layer1 component1, Loc1 layer1 component2, ...
                        Loc1 layer1 componentN, Loc1 layer2 component1, ...
                        Loc1 Layer2 componentN, ...Loc1 layerN componentN
                        Loc2 layer1 component1, ...Loc2 layerN componentN
                        LocN layer1 component1, ...LocN layerN componentN

                   3.  For complex data there Will Be 2*NDVAL data items. The
                       order is real part for value 1, imaginary part for
                       value 1, real part for value 2, imaginary part for
                       value 2, etc.              

                   4.  The order of values for various data
                       characteristics is:

                       3 DOF Global Vector: X, Y, Z
                       6 DOF Global Vector: X, Y, Z, Rx, Ry, Rz
                       Symmetric Global Tensor: Sxx, Sxy, Syy,
                                                Sxz, Syz, Szz

                       Shell and Plate Element Resultant: Fx, Fy, Fxy,
                                                          Mx, My, Mxy,
                                                          Vx, Vy

                   5.  ID line 1 always appears on plots in output
                       display.

                   6.  If result type is an "UNKNOWN" type,  id line 2
                       is displayed as data type in output display.

                   7.  Data Characteristic values (Record 9, Field 3)
                       imply the following values Of NDVALDC (Record 9,
                       Field 6)
                             Scalar:                   1
                             3 DOF Global Vector:      3
                             6 DOF Global Vector:      6
                             Symmetric Global Tensor:  6
                             General Global Tensor:    9   
                             Shell and Plate Resultant:8    
                       Since this value can also be derived from the Results
                       Type (Record 9, Field 4), this is redundant data, and
                       should be kept consistent. Some data was kept for
                       compatibility with older files.

                   8.  No entry is NOT the same as a 0. entry: all 0s must
                       be specified.

                   9.  A direct result of 8 is that if no records 14 and
                       15 appear for a node or element, this entity has
                       no data and will not be contoured, etc.

                   10. Dataloaders use the following id line convention:

                        1.   (80A1) MODEL IDENTIFICATION
                        2.   (80A1) RUN IDENTIFICATION
                        3.   (80A1) RUN DATE/TIME
                        4.   (80A1) LOAD CASE NAME
                        For static:

                        5.   (17H LOAD CASE NUMBER;, I10)

                        For normal mode:

                        5.   (10H MODE SAME, I10, 10H FREQUENCY, E13.5)

                   11. For situations with reduced # DOF'S, use 6 DOF 
                       translation and rotation with unused values = 0.

                   12. The integer associated data "number retained" will 
                       =0 unless the result set is created by sorting for 
                       extremes.  The maximum number of values to retain is 6.

        Specifed values:
          NDVAL  - Number of data values for the element. 
          NLOCS  - Number of location on the element data is stored for.
           NVALDC - Number of values for the data component.

        Derived values: 
          NLAY   - Number of location through the thickness data is stored for
                 =  NDVAL / ( NLOCS * NDVALC)
          NVLOC  - Number of values per location.
                 =  NLAY * NVALDC

        The following is always true:
        NDVAL =  NLOCS * NLAY * NVALDC

Dataset class: Data at nodes

                   1.  NLOCS = 1
                       NLAY  = 1
                    
                       NDVAL = NVALDC

                   2.  Typical fortran I/O statements for the data
                       sections are:

                             READ(LUN,1000)NUM
                             WRITE
                        1000 FORMAT (I10)
                             READ(LUN,1010) (VAL(I),I=1,NDVAL)
                             WRITE
                        1010 FORMAT (6E13.5)

                             Where: VAL is real or complex data array
                                    NUM is element number

Dataset class: Data at elements

                   1.  Data on 2D type elements may have multiple values
                       through the element thickness.  In these cases:
                           NLOCS =1
                               NLAY  =Number of layers of data through the
                                      thickness.
                       
                         NDVAL = NLAY * NVALDC

                       For solid elements: 
                         NLOCS = 1
                               NLAY  = 1
 
                         NDVAL = NVALDC

                       The order of the nodes defines an outward normal which 
                       specifies the order from position 1 to NPOS.

                   2.  Maximum Value For NVALDC Is 9.
                       No Maximum Value For NDVAL.
                       No Maximum Value For NLAY.

                   3.  Typical fortran I/O statements for the data
                       sections are:
                             READ (LUN, 1000) NUM, NDVAL
                             WRITE
                        1000 FORMAT (2I10)
                             READ (LUN, 1010) (VAL(I),I=1,NDVAL)
                             WRITE
                        1010 FORMAT (6E13.5)
 
                             Where:  VAL is real or complex data array
                                     NUM is element number
                                                                          
Dataset class: Data at nodes on elements

                   1.  Data on 2D type elements may have multiple values
                       through the element thickness.  In these cases:
                           NLOCS =Number of nodes for the element.
                               NLAY  =Number of layers of data through the
                                      thickness.
                       
                         NDVAL = NLOCS * NLAY * NVALDC

                       For solid elements: 
                         NLOCS = Number of nodes for the element.
                               NLAY  = 1
 
                         NDVAL = NLOCS * NVALDC

                       The order of the nodes defines an outward normal which 
                       specifies the order from position 1 to NPOS.    

                   2.  Maximum Value For NVALDC Is 9.
                       No Maximum Value For NDVAL.
                       No Maximum Value For NLAY.

                   3.  Typical Fortran I/O statements for the data sections
                       are:
 
                             READ (LUN,1000) NUM, IEXP, NLOCS, NVLOC
                             WRITE
                        1000 FORMAT (4I10)
                       C
                       C       Process Expansion Code 1
                       C
                             IF (IEXP.NE.1) GO TO 20
                             NSTRT = 1
                             DO 10 I=1, NLOCS
                               NSTOP = NSTRT + NVLOC - 1
                               READ (LUN,1010) (VAL(J),J=NSTRT,STOP)
                               WRITE
                        1010   FORMAT (6E13.5)
                               NSTRT = NSTRT + NVLOC
                        10   CONTINUE
                             GO TO 50
                       C
                       C       PROCESS EXPANSION CODE 2
                       C
                        20   READ (LUN,1010) (VAL(I),I=1,NVLOC)
                             NOFF = 0
                             DO 40 I=1,NLOCS
                               NOFF = NOFF +NVLOC
                               DO 30 J=1, NVLOC
                                 VAL (NOFF+J) = VAL(J)
                        30     CONTINUE
                        40   CONTINUE
                       C
                        50   NDVAL = NVLOC*NLOCS
 
                             Where:    NUM is element number. 
                                       IEXP is the element expansion code 
                                       VAL is real or complex data array. 
                                                     

Dataset class: Data at points

                   1.  Only Tetrahedral elements will be supported.

                   2.  For solid elements: 
                         NLOCS = Number of points on the element data is stored
                                 for.  Determined from the element type and 
                                 order.
                               NLAY  = 1
 
                         NDVAL = NLOCS * NVALDC

                   3.  Maximum Value For NVALDC Is 9.
                       No Maximum Value For NDVAL.

                   4.  The element order is equal to the P-order of the element

                   5.  The number of points per element is calculated from
                       the element order as follows:

                         Number_of_Points = sum(i= 1 to P-Order+1)   
                                           [sum(j = 1 to i)[1 + i - j) )]]

                   6.  Typical Fortran I/O statements for the data sections
                       are:
 
                             READ (LUN,1000) NUM, IEXP, NLOCS, NVLOC, IORDER
                             WRITE
                        1000 FORMAT (4I10)
                                          .                               
                                          .                               
                                          .                           
                           (See 3. for Data at Nodes on Elements)

                                        Analysis Type

                                                                          
  
                                                                           S
                                                                           t
                                                                           a
                                              C                       C    t
                                              o           F           o    i
                                        N     m           r           m    c
                                        o     p           e           p     
                                        r     l           q           l    N
                                        m     e     T                 e    o
                                        a     x     r     R     B     x    n
                            U           l           a     e     u     
                            n     S           E     n     s     c     E    L
                            k     t     M     i     s     p     k     i    i
                            n     a     o     g     i     o     l     g    n
                            o     t     d     e     e     n     i     e    e
                            w     i     e     n     n     s     n     n    a
                            n     c     s     1     t     e     g     2    r
                                                                      
       Design set ID        X     X     X     X     X     X     X     X    X
                                                                      
       Iteration number           X     X                             
                                                                      
       Solution set ID      X     X     X     X     X     X     X     X    X
 I                                                                    
 N     Boundary condition   X     X     X     X     X     X     X     X    X
 T                                                                    
 E     Load set                   X           X     X     X     X     X
 G                                                                    
 E     Mode number                      X     X                 X     X
 R                                                                    
       Time step number                             X                      X
                                                                      
       Frequency number                                   X           
                                                                      
       Creation option      X     X     X     X     X     X     X     X    X

       Number retained      X     X     X     X     X     X     X     X    X
         

                                                                          
    
-----------------------------------------------------------------------

       Time                                         X                      X
                                                                      
       Frequency                        X                 X           
                                                                      
       Eigenvalue                                               X     
                                                                      
       Modal Mass                       X                             
                                                                      
       Viscous damping                  X                             
                                                                      
       Hysteretic damping               X                             
                                                                      
 R     Real part eigenvalue                   X                       X
 E                                                                    
 A     Imaginary part eingenvalue             X                       X
 L                                                                    
       Real part of modal A                   X                        
       Real part of mass                                              X
                                                                      
       Imaginary part of modal A              X                      
       Imaginary part of mas                                          X
                                                                      
       Real part of modal B                   X                        
       Real part of stiffnes                                          X

       Imaginary part of modal B              X                      
       Imaginary part of stiffness                                    X
                                                                          
    
-----------------------------------------------------------------------
"""

    if raw:
        return out
    else:
        print(out)   

def _write2414(fh, dset):
    """
    DS2414_num is iterative number for each DS2414
    Nthfreq is th frequency
    Writes data at nodes - data-set 2414 - to an open file fh. 
    Currently:
       - frequency response (5) analyses are supported.
       """
    try:
        # Handle general optional fields
        dict = {'record10_field1': 0, 'record10_field2': 0, 'record10_field3': 0, 'record10_field4': 0,
                'record10_field5': 0, 'record10_field6': 0, 'record10_field7': 0, 'record10_field8': 0,
                'record11_field1': 0, 'record11_field2': 0, 'record11_field3': 0, 'record11_field4': 0,
                'record11_field5': 0, 'record11_field6': 0, 'record11_field7': 0, 'record11_field8': 0,
                'record12_field1': 0, 'record12_field2': 0, 'record12_field3': 0,
                'record12_field4': 0, 'record12_field5': 0, 'record12_field6': 0,
                'record13_field1': 0, 'record13_field2': 0, 'record13_field3': 0,
                'record13_field4': 0, 'record13_field5': 0, 'record13_field6': 0}
        dset = _opt_fields(dset, dict)
        fh.write('%6i\n%6i\n' % (-1, 2414))

        # Write general fields
        fh.write('%10i\n' % (dset['analysis_dataset_label'])) #Loadcase number (DS2414_num)
        fh.write('%-80s\n' % (dset['analysis_dataset_name'])) #usually with the frequency
        fh.write('%10i\n' % (dset['dataset_location']))
        fh.write('%-80s\n' % dset['id1'])
        fh.write('%-80s\n' % dset['id2'])
        fh.write('%-80s\n' % dset['id3']) #usually with the frequency
        fh.write('%-80s\n' % dset['id4']) #usually with the loadcase
        fh.write('%-80s\n' % dset['id5'])

        fh.write('%10i%10i%10i%10i%10i%10i\n' % (
                                        dset['model_type'], 
                                        dset['analysis_type'], 
                                        dset['data_characteristic'], 
                                        dset['result_type'],
                                        dset['data_type'], 
                                        dset['number_of_data_values_for_the_data_component']))

        # If analysis_type is 5, then give dictionairy keys a specific name
        if dset['analysis_type'] == 5:
            fh.write('%10i%10i%10i%10i%10i%10i%10i%10i\n' % (
                                            dset['design_set_id'], 
                                            dset['iteration_number'],
                                            dset['solution_set_id'],
                                            dset['boundary_condition'], 
                                            dset['load_set'],
                                            dset['mode_number'], 
                                            dset['time_step_number'],
                                            dset['frequency_number']))
            fh.write('%10i%10i\n' % (
                                            dset['creation_option'], 
                                            dset['number_retained']))
            fh.write('  %.5e  %.5e  %.5e  %.5e  %.5e  %.5e\n' % (
                                            dset['time'], 
                                            dset['frequency'], 
                                            dset['eigenvalue'], 
                                            dset['modal_mass'],
                                            dset['viscous_damping'], 
                                            dset['hysteretic_damping']))
            fh.write('  %.5e  %.5e  %.5e  %.5e  %.5e  %.5e\n' % (
                                            dset['real_part_eigenvalue'], 
                                            dset['imaginary_part_eigenvalue'], 
                                            dset['real_part_of_modal_A_or_modal_mass'], 
                                            dset['imaginary_part_of_modal_A_or_modal_mass'],
                                            dset['real_part_of_modal_B_or_modal_mass'], 
                                            dset['imaginary_part_of_modal_B_or_modal_mass']))                            
            for index in range(dset['node_nums'].shape[0]):
                fh.write('%10i\n' % (int(dset['node_nums'][index])))
                if dset['number_of_data_values_for_the_data_component'] == 3 : 
                    fh.write('%13.5e%13.5e%13.5e%13.5e%13.5e%13.5e\n' % (
                                                np.real(dset['x'][index]),
                                                np.imag(dset['x'][index]),
                                                np.real(dset['y'][index]),
                                                np.imag(dset['y'][index]),
                                                np.real(dset['z'][index]),
                                                np.imag(dset['z'][index])))
                elif dset['number_of_data_values_for_the_data_component'] == 2 : 
                    fh.write('%13.5e%13.5e%13.5e%13.5e\n' % (
                                                np.real(dset['x'][index]),
                                                np.imag(dset['x'][index]),
                                                np.real(dset['y'][index]),
                                                np.imag(dset['y'][index]),))
                elif dset['number_of_data_values_for_the_data_component'] == 1 : 
                    fh.write('%13.5e%13.5e\n' % (
                                                np.real(dset['x'][index]),
                                                np.imag(dset['x'][index]),))
                
            fh.write('%6i\n' % (-1))
        # Dictionairy keys have a general name.
        # The meaning of these records and fields depends on the software which writes/reads these.
        else:
            fh.write('%10i%10i%10i%10i%10i%10i%10i%10i\n' % (
                                            dset['record10_field1'], 
                                            dset['record10_field2'],
                                            dset['record10_field3'],
                                            dset['record10_field4'], 
                                            dset['record10_field5'],
                                            dset['record10_field6'], 
                                            dset['record10_field7'],
                                            dset['record10_field8']))
            fh.write('%10i%10i%10i%10i%10i%10i%10i%10i\n' % (
                                            dset['record11_field1'], 
                                            dset['record11_field2'],
                                            dset['record11_field3'],
                                            dset['record11_field4'], 
                                            dset['record11_field5'],
                                            dset['record11_field6'], 
                                            dset['record11_field7'],
                                            dset['record11_field8']))
            fh.write('  %.5e  %.5e  %.5e  %.5e  %.5e  %.5e\n' % (
                                            dset['record12_field1'], 
                                            dset['record12_field2'],
                                            dset['record12_field3'],
                                            dset['record12_field4'], 
                                            dset['record12_field5'],
                                            dset['record12_field6']))
            fh.write('  %.5e  %.5e  %.5e  %.5e  %.5e  %.5e\n' % (
                                            dset['record13_field1'], 
                                            dset['record13_field2'],
                                            dset['record13_field3'],
                                            dset['record13_field4'], 
                                            dset['record13_field5'],
                                            dset['record13_field6']))                        
            
            # The data and structure of Record 14 and 15 depend on the 'dataset_location'
            if dset['dataset_location'] == 1:
                for index in range(dset['node_nums'].shape[0]):
                    fh.write('%10i\n' % (dset['node_nums'][index]))
                    for field in dset['data_at_node'][index]:
                        # number of values unknown, so loop
                        fh.write('%13.5e' % (field))
                    fh.write('\n')
                fh.write('%6i\n' % (-1))

            elif dset['dataset_location'] == 2:
                for index in range(dset['element_nums'].shape[0]):
                    # get NDVAL from the datalength instead of the dictionary entry
                    fh.write('%10i%10i\n' % (dset['element_nums'][index], len(dset['data_at_element'][index])))
                    for field in dset['data_at_element'][index]:
                        # number of values in record15 unknown, so loop
                        fh.write('%13.5e' % (field))
                    fh.write('\n')
                fh.write('%6i\n' % (-1))

            elif dset['dataset_location'] == 3:
                for index in range(dset['element_nums'].shape[0]):
                    fh.write('%10i%10i%10i%10i\n' % (dset['element_nums'][index], dset['IEXP'][index],
                                                    dset['number_of_nodes'][index], dset['number_of_values_per_node'][index]))
                    for line in dset['data_at_nodes_on_element'][index]:
                        # each line is an np.ndarray with unknown number of elements, so loop
                        for item in line:
                            fh.write('%13.5e' % (item))
                        fh.write('\n')
                fh.write('%6i\n' % (-1))

            elif dset['dataset_location'] == 5:
                print('Dataset location 5 has not been implemented yet for writing dataset 2414.')

            else:
                print('Unknown dataset location ' + str(dset['dataset_location']) + '.')
                fh.write('%6i\n' % (-1))
    except Exception as msg:
        raise Exception('Error writing data-set #2414')


def _extract2414(block_data):
    """Extract analysis data - data-set 2414."""
    dset = {'type': 2414}
    # Read data
    try:
        # Processing named records and fields
        binary = False
        split_header = block_data.splitlines(True)[:15]  # Keep the line breaks!
        dset.update(_parse_header_line(split_header[2], 1, [80], [2], ['analysis_dataset_label'])) #Loadcase number
        dset.update(_parse_header_line(split_header[3], 1, [80], [1], ['analysis_dataset_name'])) # usually with the frequency
        dset.update(_parse_header_line(split_header[4], 1, [80], [2], ['dataset_location']))
        dset.update(_parse_header_line(split_header[5], 1, [80], [1], ['id1']))
        dset.update(_parse_header_line(split_header[6], 1, [80], [1], ['id2']))
        dset.update(_parse_header_line(split_header[7], 1, [80], [1], ['id3']))  # usually with the frequency
        dset.update(_parse_header_line(split_header[8], 1, [80], [1], ['id4']))  # usually with the loadcase
        dset.update(_parse_header_line(split_header[9], 1, [80], [1], ['id5']))
        
        dset.update(_parse_header_line(split_header[10], 6, [10, 10, 10, 10, 10, 10], [2, 2, 2, 2, 2, 2],
                                            ['model_type', 'analysis_type', 'data_characteristic', 'result_type',
                                                'data_type', 'number_of_data_values_for_the_data_component']))     
        
        # Processing unnamed records and fields
        # If analysis_type is 5, then give dictionairy keys a specific name
        # and process the data in record 14 and 15 following a specific pattern.
        if dset['analysis_type'] == 5:
            # frequency response specific reading functionality
            dset.update(_parse_header_line(split_header[11], 8, [10, 10, 10, 10, 10, 10, 10, 10], [2, 2, 2, 2, 2, 2, 2, 2],
                                                ['design_set_id', 'iteration_number', 'solution_set_id', 'boundary_condition', 
                                                    'load_set', 'mode_number', 'time_step_number', 'frequency_number']))
            dset.update(_parse_header_line(split_header[12], 2, [10, 10], [2, 2],
                                                ['creation_option', 'number_retained']))

            dset.update(_parse_header_line(split_header[13], 6, [13, 13, 13, 13, 13, 13], [0.5,0.5, 0.5, 0.5, 0.5, 0.5],
                                            ['time', 'frequency', 'eigenvalue', 'modal_mass', 'viscous_damping', 'hysteretic_damping']))
            dset.update(_parse_header_line(split_header[14], 6, [13, 13, 13, 13, 13, 13], [0.5,0.5, 0.5, 0.5, 0.5, 0.5],
                                            ['real_part_eigenvalue', 'imaginary_part_eigenvalue', 
                                                'real_part_of_modal_A_or_modal_mass', 'imaginary_part_of_modal_A_or_modal_mass', 
                                                'real_part_of_modal_B_or_modal_mass', 'imaginary_part_of_modal_B_or_modal_mass']))

            split_data = ''.join(block_data.splitlines(True)[15:])
            split_data = split_data.split()
            # generic reading method
            n_skip = dset['number_of_data_values_for_the_data_component'] * 2 + 1 # x2 real/imag
            
            if dset['data_type'] == 5 :
                values = np.asarray([float(str) for str in split_data], 'd')
                dset['node_nums'] = np.array(values[::n_skip].copy(), dtype=int)
                
                dset['x'] = values[1::n_skip].copy()+values[2::n_skip].copy()*1j
                
                if dset['number_of_data_values_for_the_data_component'] >= 2 :
                    dset['y'] = values[3::n_skip].copy()+values[4::n_skip].copy()*1j
                
                if dset['number_of_data_values_for_the_data_component'] >= 3 :
                    dset['z'] = values[5::n_skip].copy()+values[6::n_skip].copy()*1j   

        # Processing unnamed records and fields
        # Dictionairy keys have a general name.
        # The meaning of these records and fields depends on the software which writes/reads these.
        # This section processes the data in record 14 and 15 as general as possible.
        # general reading functionality min_values is set to 1 in _parse_header_line to keep it as general as possible
        else:
            dset.update(_parse_header_line(split_header[11], 1, [10, 10, 10, 10, 10, 10, 10, 10], [2, 2, 2, 2, 2, 2, 2, 2],
                                                ['record10_field1', 'record10_field2', 'record10_field3', 'record10_field4', 
                                                    'record10_field5', 'record10_field6', 'record10_field7', 'record10_field8']))
            dset.update(_parse_header_line(split_header[12], 1, [10, 10, 10, 10, 10, 10, 10, 10], [2, 2, 2, 2, 2, 2, 2, 2],
                                                ['record11_field1', 'record11_field2', 'record11_field3', 'record11_field4', 
                                                    'record11_field5', 'record11_field6', 'record11_field7', 'record11_field8']))

            dset.update(_parse_header_line(split_header[13], 1, [13, 13, 13, 13, 13, 13], [0.5,0.5, 0.5, 0.5, 0.5, 0.5],
                                            ['record12_field1', 'record12_field2', 'record12_field3', 'record12_field4',
                                            'record12_field5', 'record12_field6']))
            dset.update(_parse_header_line(split_header[14], 1, [13, 13, 13, 13, 13, 13], [0.5,0.5, 0.5, 0.5, 0.5, 0.5],
                                            ['record13_field1', 'record13_field2', 'record13_field3', 'record13_field4',
                                            'record13_field5', 'record13_field6']))
            
            # The data and structure of Record 14 and 15 depend on the 'dataset_location'
            split_data = ''.join(block_data.splitlines(True)[15:])
            split_data = split_data.splitlines()
            if dset['dataset_location'] == 1:
                # Data at nodes
                record14 = np.asarray([str.split() for str in split_data[0::2]], 'U')
                # record 15 not always same number of fields, so a list of ndarray
                record15 = [np.array([float(elem) for elem in item.split()], float) for item in split_data[1::2]]
                dset['node_nums'] = np.array(record14[:,0].copy(), dtype=int)
                dset['data_at_node'] = record15

            elif dset['dataset_location'] == 2:
                # Data on elements
                record14 = np.asarray([str.split() for str in split_data[0::2]], 'U')
                ## record 15 not always same number of fields, so a list of ndarray
                record15 = [np.array([float(elem) for elem in item.split()], float) for item in split_data[1::2]]
                dset['element_nums'] = np.array(record14[:,0].copy(), dtype=int)
                dset['NDVAL'] = np.array(record14[:,1].copy(), dtype=int)
                dset['data_at_element'] = record15

            elif dset['dataset_location'] == 3:
                # Data at nodes on elements
                # This one is special, as record15 can be repeated multiple times
                # if data is present for all nodes (IEXP=1)
                # therefore we need to loop
                dset['element_nums'] = []
                dset['IEXP'] = []
                dset['number_of_nodes'] = []
                dset['number_of_values_per_node'] = []
                dset['data_at_nodes_on_element'] = []
                lineIndex = 0
                while lineIndex < len(split_data):
                    dset['element_nums'].append(int(split_data[lineIndex].split()[0]))
                    iEXP = int(split_data[lineIndex].split()[1])
                    dset['IEXP'].append(iEXP)
                    numberOfNodes = int(split_data[lineIndex].split()[2])
                    dset['number_of_nodes'].append(numberOfNodes)
                    numberOfValuesPerNode = int(split_data[lineIndex].split()[3])
                    dset['number_of_values_per_node'].append(numberOfValuesPerNode)
                    # if IEXP = 2, only one line for all nodes. Thus floor division + 1
                    numberOfLinesInRecord15 = 0
                    if iEXP == 1:
                        numberOfLinesInRecord15 = numberOfNodes * math.ceil(numberOfValuesPerNode / 6)
                    else:
                        numberOfLinesInRecord15 = math.ceil(numberOfValuesPerNode / 6)
                    # a list of numpy array, since the number of lines is not fixed.
                    dset['data_at_nodes_on_element'].append([np.array([float(elem) for elem in item.split()], float) for item in split_data[lineIndex + 1:lineIndex + 1 + numberOfLinesInRecord15]])

                    lineIndex = lineIndex + 1 + numberOfLinesInRecord15

                # make the lists into array, for easier handling the results.
                dset['element_nums'] = np.array(dset['element_nums'], int)
                dset['IEXP'] = np.array(dset['IEXP'], int)
                dset['number_of_nodes'] = np.array(dset['number_of_nodes'], int)
                dset['number_of_values_per_node'] = np.array(dset['number_of_values_per_node'], int)

            elif dset['dataset_location'] == 5:
                print('Dataset location 5 has not been implemented yet for reading dataset 2414.')

            else:
                print('Dataset location ' + str(dset['dataset_location']) + 'not supported')
                pass

    except:
        raise Exception('Error reading data-set #2414')
    return dset


def prepare_2414(
        analysis_dataset_label=None,
        analysis_dataset_name=None,
        dataset_location=None,
        id1=None,
        id2=None,
        id3=None,
        id4=None,
        id5=None,
        model_type=None,
        analysis_type=None,
        data_characteristic=None,
        result_type=None,
        data_type=None,
        number_of_data_values_for_the_data_component=None,
        design_set_id=None,
        iteration_number=None,
        solution_set_id=None,
        boundary_condition=None,
        load_set=None,
        mode_number=None,
        time_step_number=None,
        frequency_number=None,
        creation_option=None,
        number_retained=None,
        time=None,
        frequency=None,
        eigenvalue=None,
        modal_mass=None,
        viscous_damping=None,
        hysteretic_damping=None,
        real_part_eigenvalue=None,
        imaginary_part_eigenvalue=None,
        real_part_of_modal_A_or_modal_mass=None,
        imaginary_part_of_modal_A_or_modal_mass=None,
        real_part_of_modal_B_or_modal_mass=None,
        imaginary_part_of_modal_B_or_modal_mass=None,
        d=None,
        node_nums=None,
        x=None,
        y=None,
        z=None,
        return_full_dict=False):
    """Name: Analysis Data

    R-Record, F-Field

    :param analysis_dataset_label: R1 F1, Analysis dataset label
    :param analysis_dataset_name: R2 F1, Analysis dataset name 
    :param dataset_location: R3 F1, Dataset location, 1:Data at nodes, 2:Data on elements, 3:Data at nodes on elements, 5:Data at points

    :param id1: R4 F1, ID line 1
    :param id2: R5 F1, ID line 2
    :param id3: R6 F1, ID line 3
    :param id4: R7 F1, ID line 4
    :param id5: R8 F1, ID line 5
    :param model_type: R9 F1, Model type
    :param analysis_type: R9 F2, Analysis type 
    :param data_characteristic: R9 F3, Data characteristic
    :param result_type: R9 F4, Result type
    :param data_type: R9 F5, Data type
    :param number_of_data_values_for_the_data_component: R9 F6,  Number of data values for the data component (NVALDC)
    
    **Integer analysis type specific data**
    
    :param design_set_id: R10 F1, 
    :param iteration_number: R10 F1,
    :param solution_set_id: R10 F1,
    :param boundary_condition: R10 F1,
    :param load_set: R10 F1,
    :param mode_number: R10 F1,
    :param time_step_number: R10 F1,
    :param frequency_number: R10 F1,
    
    :param creation_option: R11 F1,
    :param number_retained: R11 F1,

    **Real analysis type specific data**

    :param time: R12 F1,
    :param frequency: R12 F1,
    :param eigenvalue: R12 F1,
    :param modal_mass: R12 F1,
    :param viscous_damping: R12 F1,
    :param hysteretic_damping: R12 F1,
    
    :param real_part_eigenvalue: R13 F1,
    :param imaginary_part_eigenvalue: R13 F1,
    :param real_part_of_modal_A_or_modal_mass: R13 F1,
    :param imaginary_part_of_modal_A_or_modal_mass: R13 F1,
    :param real_part_of_modal_B_or_modal_mass,: R13 F1,
    :param imaginary_part_of_modal_B_or_modal_mass: R13 F1,
    :param node_nums: R14 F1, Node number
    :param d: R15 F1, Data at this node (NDVAL real or complex values)
    :param x: R15 F1,
    :param y: R15 F1,
    :param z: R15 F1,

    :param return_full_dict: If True full dict with all keys is returned, else only specified arguments are included

    **Test prepare_2414**

    >>> save_to_file = 'test_pyuff'
    >>> dataset = pyuff.prepare_2414(
    >>>     analysis_dataset_label=1,
    >>>     analysis_dataset_name='Solid displacement at 1.00000e+02 Hz',
    >>>     dataset_location=1,
    >>>     id1='FEMTown to UNV output driver',
    >>>     id2=' Analysis',
    >>>     id3='Solid displacement at 1.00000e+02 Hz',
    >>>     id4='LoadCase 1',
    >>>     id5='none',
    >>>     model_type=0,
    >>>     analysis_type=5,
    >>>     data_characteristic=2,
    >>>     result_type=8,
    >>>     data_type=5,
    >>>     number_of_data_values_for_the_data_component=3,
    >>>     design_set_id=1,
    >>>     iteration_number=0,
    >>>     solution_set_id=1,
    >>>     boundary_condition=0,
    >>>     load_set=1,
    >>>     mode_number=0,
    >>>     time_step_number=0,
    >>>     frequency_number=0,
    >>>     creation_option=1,
    >>>     number_retained=1,
    >>>     time=0.0,
    >>>     frequency=100.0,
    >>>     eigenvalue=100.0,
    >>>     modal_mass=0.0,
    >>>     viscous_damping=0.0,
    >>>     hysteretic_damping=0.0,
    >>>     real_part_eigenvalue=0.0,
    >>>     imaginary_part_eigenvalue=0.0,
    >>>     real_part_of_modal_A_or_modal_mass=0.0,
    >>>     imaginary_part_of_modal_A_or_modal_mass=0.0,
    >>>     real_part_of_modal_B_or_modal_mass=0.0,
    >>>     imaginary_part_of_modal_B_or_modal_mass=0.0,
    >>>     node_nums=np.array([29, 30, 31]),
    >>>     x=np.array([2.84811e-09+1.73733e-09j, 3.11873e-09+1.19904e-09j, 3.28182e-09+6.12300e-10j]),
    >>>     y=np.array([-7.13813e-10+6.53419e-10j, -4.95859e-10+8.07390e-10j, -2.52772e-10+9.05991e-10j]),
    >>>     z=np.array([4.28828e-10+2.47563e-10j, 4.65334e-10+1.70107e-10j, 4.87383e-10+8.66597e-11j]))
    >>> if save_to_file:
    >>>     if os.path.exists(save_to_file):
    >>>         os.remove(save_to_file)
    >>>     uffwrite = pyuff.UFF(save_to_file)
    >>>     uffwrite.write_sets(dataset, mode='add')
    >>> dataset
    """
    if np.array(analysis_dataset_label).dtype != int and analysis_dataset_label != None:
        raise TypeError('analysis_dataset_label must be integer')
    if type(analysis_dataset_name) != str and analysis_dataset_name != None:
         raise TypeError('analysis_dataset_name must be string')
    if dataset_location not in (1, 2, 3, 5, None):
        raise ValueError('dataset_location can be: 1, 2, 3, 5')
    if type(id1) != str and id1 != None:
         raise TypeError('id1 must be string')
    if type(id2) != str and id2 != None:
         raise TypeError('id2 must be string')
    if type(id3) != str and id3 != None:
         raise TypeError('id3 must be string')
    if type(id4) != str and id4 != None:
         raise TypeError('id4 must be string')    
    if type(id5) != str and id5 != None:
         raise TypeError('id5 must be string')

    if model_type not in (0, 1, 2, 3, None):
        raise ValueError('model_type can be: 0, 1, 2, 3')
    if analysis_type not in (0, 1, 2, 3, 4, 5, 6, 7, 9, None):
        raise ValueError('analysis_type can be: 0, 1, 2, 3, 4, 6, 7, 9')
    if data_characteristic not in (0, 1, 2, 3, 4, 6, None):
        raise ValueError('data_characteristic can be: 0, 1, 2, 3, 4, 6')
    if type(result_type) != int and result_type != None:
        raise TypeError('result type must be integer')
    if data_type not in (1, 2, 4, 5, 6, None):
        raise ValueError('data_characteristic can be: 1, 2, 4, 5, 6')
    if np.array(number_of_data_values_for_the_data_component).dtype != int and number_of_data_values_for_the_data_component != None:
        raise TypeError('number_of_data_values_for_the_data_component type must be integer')
    if np.array(design_set_id).dtype != int and design_set_id != None:
        raise TypeError('design_set_id must be integer')
    if np.array(iteration_number).dtype != int and iteration_number != None:
        raise TypeError('iteration_number must be integer')
    if np.array(solution_set_id).dtype != int and solution_set_id != None:
        raise TypeError('solution_set_id must be integer')
    if np.array(boundary_condition).dtype != int and boundary_condition != None:
        raise TypeError('boundary_condition must be integer')
    if np.array(load_set).dtype != int and load_set != None:
        raise TypeError('load_set must be integer')
    if np.array(mode_number).dtype != int and mode_number != None:
        raise TypeError('mode_number must be integer')
    if np.array(time_step_number).dtype != int and time_step_number != None:
        raise TypeError('time_step_number must be integer')
    if np.array(frequency_number).dtype != int and frequency_number != None:
        raise TypeError('frequency_number must be integer')
    
    if np.array(creation_option).dtype != int and creation_option != None:
        raise TypeError('creation_option must be integer')
    if np.array(number_retained).dtype != int and number_retained != None:
        raise TypeError('number_retained must be integer')
    
    if np.array(time).dtype != float and time != None:
        raise TypeError('time must be float')
    if np.array(frequency).dtype != float and frequency != None:
        raise TypeError('frequency must be float')
    if np.array(eigenvalue).dtype != float and eigenvalue != None:
        raise TypeError('eigenvalue must be float')
    if np.array(modal_mass).dtype != float and modal_mass != None:
        raise TypeError('modal_mass must be float')
    if np.array(viscous_damping).dtype != float and viscous_damping != None:
        raise TypeError('viscous_damping must be float')
    if np.array(hysteretic_damping).dtype != float and hysteretic_damping != None:
        raise TypeError('hysteretic_damping must be float')
    if np.array(real_part_eigenvalue).dtype != float and real_part_eigenvalue != None:
        raise TypeError('real_part_eigenvalue must be float')
    if np.array(imaginary_part_eigenvalue).dtype != float and imaginary_part_eigenvalue != None:
        raise TypeError('imaginary_part_eigenvalue must be float')
    if np.array(real_part_of_modal_A_or_modal_mass).dtype != float and real_part_of_modal_A_or_modal_mass != None:
        raise TypeError('real_part_of_modal_A_or_modal_mass must be float')
    if np.array(imaginary_part_of_modal_A_or_modal_mass).dtype != float and imaginary_part_of_modal_A_or_modal_mass != None:
        raise TypeError('imaginary_part_of_modal_A_or_modal_mass must be float')
    if np.array(real_part_of_modal_B_or_modal_mass).dtype != float and real_part_of_modal_B_or_modal_mass != None:
        raise TypeError('real_part_of_modal_B_or_modal_mass must be float')
    if np.array(imaginary_part_of_modal_B_or_modal_mass).dtype != float and imaginary_part_of_modal_B_or_modal_mass != None:
        raise TypeError('imaginary_part_of_modal_B_or_modal_mass must be float')
    
    if np.array(node_nums).dtype != int and node_nums != None:
        raise TypeError('node_nums must be integer')
    if np.array(d).dtype != float and d != None:
        raise TypeError('d must be float')
    if np.array(x).dtype != float and np.array(x).dtype != complex and x != None:
        raise TypeError('x must be float')
    if np.array(y).dtype != float and np.array(y).dtype != complex and y != None:
        raise TypeError('y must be float')
    if np.array(z).dtype != float and np.array(z).dtype != complex and z != None:
        raise TypeError('z must be float')

    dataset={
        'type': 2414,
        'analysis_dataset_label': analysis_dataset_label,
        'analysis_dataset_name': analysis_dataset_name,
        'dataset_location': dataset_location,
        'id1': id1,
        'id2': id2,
        'id3': id3,
        'id4': id4,
        'id5': id5,
        'model_type': model_type,
        'analysis_type': analysis_type,
        'data_characteristic': data_characteristic,
        'result_type': result_type,
        'data_type': data_type,
        'number_of_data_values_for_the_data_component': number_of_data_values_for_the_data_component,
        'design_set_id': design_set_id,
        'iteration_number': iteration_number,
        'solution_set_id': solution_set_id,
        'boundary_condition': boundary_condition,
        'load_set': load_set,
        'mode_number': mode_number,
        'time_step_number': time_step_number,
        'frequency_number': frequency_number,
        'creation_option': creation_option,
        'number_retained': number_retained,
        'time': time,
        'frequency': frequency,
        'eigenvalue': eigenvalue,
        'modal_mass': modal_mass,
        'viscous_damping': viscous_damping,
        'hysteretic_damping': hysteretic_damping,
        'real_part_eigenvalue': real_part_eigenvalue,
        'imaginary_part_eigenvalue': imaginary_part_eigenvalue,
        'real_part_of_modal_A_or_modal_mass': real_part_of_modal_A_or_modal_mass, 
        'imaginary_part_of_modal_A_or_modal_mass': imaginary_part_of_modal_A_or_modal_mass,
        'real_part_of_modal_B_or_modal_mass': real_part_of_modal_B_or_modal_mass, 
        'imaginary_part_of_modal_B_or_modal_mass': imaginary_part_of_modal_B_or_modal_mass,
        'd': d,
        'node_nums': node_nums,
        'x': x,
        'y': y,
        'z': z
        }

    if return_full_dict is False:
        dataset = check_dict_for_none(dataset)

    return dataset

