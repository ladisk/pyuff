import numpy as np
import struct
import sys
import os

from ..tools import _opt_fields, _parse_header_line, check_dict_for_none

def get_structure_58(raw=False):
    """(source: https://www.ceas3.uc.edu/sdrluff/"""
    out = """
Universal Dataset Number: 58

Name:   Function at Nodal DOF
-----------------------------------------------------------------------
 
         Record 1:     Format(80A1)
                       Field 1    - ID Line 1
 
                                                 NOTE
 
                           ID Line 1 is generally  used  for  the function
                           description.
 
 
         Record 2:     Format(80A1)
                       Field 1    - ID Line 2
 
         Record 3:     Format(80A1)
                       Field 1    - ID Line 3
 
                                                 NOTE
 
                           ID Line 3 is generally used to identify when the
                           function  was  created.  The date is in the form
                           DD-MMM-YY, and the time is in the form HH:MM:SS,
                           with a general Format(9A1,1X,8A1).
 
 
         Record 4:     Format(80A1)
                       Field 1    - ID Line 4
 
         Record 5:     Format(80A1)
                       Field 1    - ID Line 5
 
         Record 6:     Format(2(I5,I10),2(1X,10A1,I10,I4))
                                  DOF Identification
                       Field 1    - Function Type
                                    0 - General or Unknown
                                    1 - Time Response
                                    2 - Auto Spectrum
                                    3 - Cross Spectrum
                                    4 - Frequency Response Function
                                    5 - Transmissibility
                                    6 - Coherence
                                    7 - Auto Correlation
                                    8 - Cross Correlation
                                    9 - Power Spectral Density (PSD)
                                    10 - Energy Spectral Density (ESD)
                                    11 - Probability Density Function
                                    12 - Spectrum
                                    13 - Cumulative Frequency Distribution
                                    14 - Peaks Valley
                                    15 - Stress/Cycles
                                    16 - Strain/Cycles
                                    17 - Orbit
                                    18 - Mode Indicator Function
                                    19 - Force Pattern
                                    20 - Partial Power
                                    21 - Partial Coherence
                                    22 - Eigenvalue
                                    23 - Eigenvector
                                    24 - Shock Response Spectrum
                                    25 - Finite Impulse Response Filter
                                    26 - Multiple Coherence
                                    27 - Order Function
                       Field 2    - Function Identification Number
                       Field 3    - Version Number, or sequence number
                       Field 4    - Load Case Identification Number
                                    0 - Single Point Excitation
                       Field 5    - Response Entity Name ("NONE" if unused)
                       Field 6    - Response Node
                       Field 7    - Response Direction
                                     0 - Scalar
                                     1 - +X Translation       4 - +X Rotation
                                    -1 - -X Translation      -4 - -X Rotation
                                     2 - +Y Translation       5 - +Y Rotation
                                    -2 - -Y Translation      -5 - -Y Rotation
                                     3 - +Z Translation       6 - +Z Rotation
                                    -3 - -Z Translation      -6 - -Z Rotation
                       Field 8    - Reference Entity Name ("NONE" if unused)
                       Field 9    - Reference Node
                       Field 10   - Reference Direction  (same as field 7)
 
                                                 NOTE
 
                           Fields 8, 9, and 10 are only relevant if field 4
                           is zero.
 
 
         Record 7:     Format(3I10,3E13.5)
                                  Data Form
                       Field 1    - Ordinate Data Type
                                    2 - real, single precision
                                    4 - real, double precision
                                    5 - complex, single precision
                                    6 - complex, double precision
                       Field 2    - Number of data pairs for uneven abscissa
                                    spacing, or number of data values for even
                                    abscissa spacing
                       Field 3    - Abscissa Spacing
                                    0 - uneven
                                    1 - even (no abscissa values stored)
                       Field 4    - Abscissa minimum (0.0 if spacing uneven)
                       Field 5    - Abscissa increment (0.0 if spacing uneven)
                       Field 6    - Z-axis value (0.0 if unused)
 
         Record 8:     Format(I10,3I5,2(1X,20A1))
                                  Abscissa Data Characteristics
                       Field 1    - Specific Data Type
                                    0 - unknown
                                    1 - general
                                    2 - stress
                                    3 - strain
                                    5 - temperature
                                    6 - heat flux
                                    8 - displacement
                                    9 - reaction force
                                    11 - velocity
                                    12 - acceleration
                                    13 - excitation force
                                    15 - pressure
                                    16 - mass
                                    17 - time
                                    18 - frequency
                                    19 - rpm
                                    20 - order
                       Field 2    - Length units exponent
                       Field 3    - Force units exponent
                       Field 4    - Temperature units exponent
 
                                                 NOTE
 
                           Fields 2, 3 and  4  are  relevant  only  if the
                           Specific Data Type is General, or in the case of
                           ordinates, the response/reference direction is a
                           scalar, or the functions are being used for
                           nonlinear connectors in System Dynamics Analysis.
                           See Addendum 'A' for the units exponent table.
 
                       Field 5    - Axis label ("NONE" if not used)
                       Field 6    - Axis units label ("NONE" if not used)
 
                                                 NOTE
 
                           If fields  5  and  6  are  supplied,  they take
                           precendence  over  program  generated labels and
                           units.
 
         Record 9:     Format(I10,3I5,2(1X,20A1))
                       Ordinate (or ordinate numerator) Data Characteristics
 
         Record 10:    Format(I10,3I5,2(1X,20A1))
                       Ordinate Denominator Data Characteristics
 
         Record 11:    Format(I10,3I5,2(1X,20A1))
                       Z-axis Data Characteristics
 
                                                 NOTE
 
                           Records 9, 10, and 11 are  always  included and
                           have fields the same as record 8.  If records 10
                           and 11 are not used, set field 1 to zero.
 
         Record 12:
                                   Data Values
 
                         Ordinate            Abscissa
             Case     Type     Precision     Spacing       Format
           -------------------------------------------------------------
               1      real      single        even         6E13.5
               2      real      single       uneven        6E13.5
               3     complex    single        even         6E13.5
               4     complex    single       uneven        6E13.5
               5      real      double        even         4E20.12
               6      real      double       uneven     2(E13.5,E20.12)
               7     complex    double        even         4E20.12
               8     complex    double       uneven      E13.5,2E20.12
           --------------------------------------------------------------
 
                                          NOTE
 
           See  Addendum  'B'  for  typical  FORTRAN   READ/WRITE
           statements for each case.
 
 
         General Notes:
 
              1.  ID lines may not be blank.  If no  information  is required,
                  the word "NONE" must appear in columns 1 through 4.
 
              2.  ID line 1 appears on plots in Finite Element Modeling and is
                  used as the function description in System Dynamics Analysis.
 
              3.  Dataloaders use the following ID line conventions
                     ID Line 1 - Model Identification
                     ID Line 2 - Run Identification
                     ID Line 3 - Run Date and Time
                     ID Line 4 - Load Case Name
 
              4.  Coordinates codes from MODAL-PLUS and MODALX are decoded into
                  node and direction.
 
              5.  Entity names used in System Dynamics Analysis prior to I-DEAS
                  Level 5 have a 4 character maximum. Beginning with Level 5,
                  entity names will be ignored if this dataset is preceded by
                  dataset 259. If no dataset 259 precedes this dataset, then the
                  entity name will be assumed to exist in model bin number 1.
 
              6.  Record 10 is ignored by System Dynamics Analysis unless load
                  case = 0. Record 11 is always ignored by System Dynamics
                  Analysis.
 
              7.  In record 6, if the response or reference names are "NONE"
                  and are not overridden by a dataset 259, but the correspond-
                  ing node is non-zero, System Dynamics Analysis adds the node
                  and direction to the function description if space is sufficie
 
              8.  ID line 1 appears on XY plots in Test Data Analysis along
                  with ID line 5 if it is defined.  If defined, the axis units
                  labels also appear on the XY plot instead of the normal
                  labeling based on the data type of the function.
 
              9.  For functions used with nonlinear connectors in System
                  Dynamics Analysis, the following requirements must be
                  adhered to:
 
                  a) Record 6: For a displacement-dependent function, the
                     function type must be 0; for a frequency-dependent
                     function, it must be 4. In either case, the load case
                     identification number must be 0.
 
                  b) Record 8: For a displacement-dependent function, the
                     specific data type must be 8 and the length units
                     exponent must be 0 or 1; for a frequency-dependent
                     function, the specific data type must be 18 and the
                     length units exponent must be 0. In either case, the
                     other units exponents must be 0.
 
                  c) Record 9: The specific data type must be 13. The
                     temperature units exponent must be 0. For an ordinate
                     numerator of force, the length and force units
                     exponents must be 0 and 1, respectively. For an
                     ordinate numerator of moment, the length and force
                     units exponents must be 1 and 1, respectively.
 
                  d) Record 10: The specific data type must be 8 for
                     stiffness and hysteretic damping; it must be 11
                     for viscous damping. For an ordinate denominator of
                     translational displacement, the length units exponent
                     must be 1; for a rotational displacement, it must
                     be 0. The other units exponents must be 0.
 
                  e) Dataset 217 must precede each function in order to
                     define the function's usage (i.e. stiffness, viscous
                     damping, hysteretic damping).
 
                                       Addendum A
 
         In order to correctly perform units  conversion,  length,  force, and
         temperature  exponents  must  be  supplied for a specific data type of
         General; that is, Record 8 Field 1 = 1.  For example, if the function
         has  the  physical dimensionality of Energy (Force * Length), then the
         required exponents would be as follows:
 
                 Length = 1
                 Force = 1          Energy = L * F
                 Temperature = 0
 
         Units exponents for the remaining specific data types  should not  be
         supplied.  The following exponents will automatically be used.
 
 
                              Table - Unit Exponents
              -------------------------------------------------------
               Specific                   Direction
                        ---------------------------------------------
                 Data       Translational            Rotational
                        ---------------------------------------------
                 Type    Length  Force  Temp    Length  Force  Temp
              -------------------------------------------------------
                  0        0       0      0       0       0      0
                  1             (requires input to fields 2,3,4)
                  2       -2       1      0      -1       1      0
                  3        0       0      0       0       0      0
                  5        0       0      1       0       0      1
                  6        1       1      0       1       1      0
                  8        1       0      0       0       0      0
                  9        0       1      0       1       1      0
                 11        1       0      0       0       0      0
                 12        1       0      0       0       0      0
                 13        0       1      0       1       1      0
                 15       -2       1      0      -1       1      0
                 16       -1       1      0       1       1      0
                 17        0       0      0       0       0      0
                 18        0       0      0       0       0      0
                 19        0       0      0       0       0      0
              --------------------------------------------------------
 
                                          NOTE
 
                 Units exponents for scalar points are defined within
                 System Analysis prior to reading this dataset.
 
                                       Addendum B
 
         There are 8 distinct  combinations  of  parameters  which  affect the
         details   of  READ/WRITE  operations.   The  parameters  involved are
         Ordinate Data Type, Ordinate Data  Precision,  and  Abscissa Spacing.
         Each  combination  is documented in the examples below.  In all cases,
         the number of data values (for even abscissa spacing)  or  data pairs
         (for  uneven  abscissa  spacing)  is NVAL.  The abcissa is always real
         single precision.  Complex double precision is  handled  by  two real
         double  precision  variables  (real  part  followed by imaginary part)
         because most systems do not directly support complex double precision.
 
         CASE 1
 
         REAL
         SINGLE PRECISION
         EVEN SPACING
 
           Order of data in file           Y1   Y2   Y3   Y4   Y5   Y6
                                           Y7   Y8   Y9   Y10  Y11  Y12
                                                      .
                                                      .
                                                      .
           Input
 
                     REAL Y(6)
                       .
                       .
                       .
                     NPRO=0
                  10 READ(LUN,1000,ERR=  ,END=  )(Y(I),I=1,6)
                1000 FORMAT(6E13.5)
                     NPRO=NPRO+6
                       .
                       .    code to process these six values
                       .
                     IF(NPRO.LT.NVAL)GO TO 10
                       .
                       .   continued processing
                       .
 
           Output
 
                     REAL Y(6)
                       .
                       .
                       .
                     NPRO=0
                  10 CONTINUE
                       .
                       .    code to set up these six values
                       .
                     WRITE(LUN,1000,ERR=  )(Y(I),I=1,6)
                1000 FORMAT(6E13.5)
                     NPRO=NPRO+6
 
                     IF(NPRO.LT.NVAL)GO TO 10
                       .
                       .   continued processing
                       .
 
         CASE 2
 
         REAL
         SINGLE PRECISION
         UNEVEN SPACING
 
           Order of data in file          X1   Y1   X2   Y2   X3   Y3
                                          X4   Y4   X5   Y5   X6   Y6
                                           .
                                           .
                                           .
 
           Input
 
                     REAL X(3),Y(3)
                       .
                       .
                       .
                     NPRO=0
                  10 READ(LUN,1000,ERR=  ,END=  )(X(I),Y(I),I=1,3)
                1000 FORMAT(6E13.5)
                     NPRO=NPRO+3
                       .
                       .    code to process these three values
                       .
                     IF(NPRO.LT.NVAL)GO TO 10
                       .
                       .   continued processing
                       .
 
 
           Output
 
                     REAL X(3),Y(3)
                       .
                       .
                       .
                     NPRO=0
                  10 CONTINUE
                       .
                       .    code to set up these three values
                       .
                     WRITE(LUN,1000,ERR=  )(X(I),Y(I),I=1,3)
                1000 FORMAT(6E13.5)
                     NPRO=NPRO+3
                     IF(NPRO.LT.NVAL)GO TO 10
                       .
                       .   continued processing
                       .
 
         CASE 3
 
         COMPLEX
         SINGLE PRECISION
         EVEN SPACING
 
           Order of data in file          RY1  IY1  RY2  IY2  RY3  IY3
                                          RY4  IY4  RY5  IY5  RY6  IY6
                                           .
                                           .
                                           .
 
           Input
 
                     COMPLEX Y(3)
                       .
                       .
                       .
                     NPRO=0
                  10 READ(LUN,1000,ERR=  ,END=  )(Y(I),I=1,3)
                1000 FORMAT(6E13.5)
                     NPRO=NPRO+3
                       .
                       .    code to process these six values
                       .
                     IF(NPRO.LT.NVAL)GO TO 10
                       .
                       .   continued processing
                       .
 
 
           Output
 
                     COMPLEX Y(3)
                       .
                       .
                       .
                     NPRO=0
                  10 CONTINUE
                       .
                       .    code to set up these three values
                       .
                     WRITE(LUN,1000,ERR=  )(Y(I),I=1,3)
                1000 FORMAT(6E13.5)
                     NPRO=NPRO+3
                     IF(NPRO.LT.NVAL)GO TO 10
                       .
                       .   continued processing
                       .
 
         CASE 4
 
         COMPLEX
         SINGLE PRECISION
         UNEVEN SPACING
 
           Order of data in file          X1   RY1  IY1  X2  RY2  IY2
                                          X3   RY3  IY3  X4  RY4  IY4
 
                                           .
                                           .
                                           .
 
           Input
 
                     REAL X(2)
                     COMPLEX Y(2)
                       .
                       .
                       .
                     NPRO=0
                  10 READ(LUN,1000,ERR=  ,END=  )(X(I),Y(I),I=1,2)
                1000 FORMAT(6E13.5)
                     NPRO=NPRO+2
                       .
                       .    code to process these two values
                       .
                     IF(NPRO.LT.NVAL)GO TO 10
                       .
                       .   continued processing
                       .
 
          Output
 
                     REAL X(2)
                     COMPLEX Y(2)
                       .
                       .
                       .
                     NPRO=0
                  10 CONTINUE
                       .
                       .    code to set up these two values
                       .
                     WRITE(LUN,1000,ERR=  )(X(I),Y(I),I=1,2)
                1000 FORMAT(6E13.5)
                     NPRO=NPRO+2
                     IF(NPRO.LT.NVAL)GO TO 10
                       .
                       .   continued processing
                       .
 
         CASE 5
 
         REAL
         DOUBLE PRECISION
         EVEN SPACING
 
           Order of data in file          Y1     Y2     Y3     Y4
                                          Y5     Y6     Y7     Y8
                                           .
                                           .
                                           .
           Input
 
                     DOUBLE PRECISION Y(4)
                       .
                       .
                       .
                     NPRO=0
                  10 READ(LUN,1000,ERR=  ,END=  )(Y(I),I=1,4)
                1000 FORMAT(4E20.12)
                     NPRO=NPRO+4
                       .
                       .    code to process these four values
                       .
                     IF(NPRO.LT.NVAL)GO TO 10
                       .
                       .   continued processing
                       .
 
 
           Output
 
                     DOUBLE PRECISION Y(4)
                       .
                       .
                       .
                     NPRO=0
                  10 CONTINUE
                       .
                       .    code to set up these four values
                       .
                     WRITE(LUN,1000,ERR=  )(Y(I),I=1,4)
                1000 FORMAT(4E20.12)
                     NPRO=NPRO+4
                     IF(NPRO.LT.NVAL)GO TO 10
                       .
                       .   continued processing
                       .
 
         CASE 6
 
         REAL
         DOUBLE PRECISION
         UNEVEN SPACING
 
           Order of data in file          X1   Y1     X2   Y2
                                          X3   Y3     X4   Y4
                                           .
                                           .
                                           .
           Input
 
                     REAL X(2)
                     DOUBLE PRECISION Y(2)
                       .
                       .
                       .
                     NPRO=0
                  10 READ(LUN,1000,ERR=  ,END=  )(X(I),Y(I),I=1,2)
                1000 FORMAT(2(E13.5,E20.12))
                     NPRO=NPRO+2
                       .
                       .    code to process these two values
                       .
                     IF(NPRO.LT.NVAL)GO TO 10
                       .
                       .   continued processing
                       .
 
           Output
 
                     REAL X(2)
                     DOUBLE PRECISION Y(2)
                       .
                       .
                       .
                     NPRO=0
                  10 CONTINUE
                       .
                       .    code to set up these two values
                       .
                     WRITE(LUN,1000,ERR=  )(X(I),Y(I),I=1,2)
                1000 FORMAT(2(E13.5,E20.12))
                     NPRO=NPRO+2
                     IF(NPRO.LT.NVAL)GO TO 10
                       .
                       .   continued processing
                       .
 
         CASE 7
 
         COMPLEX
         DOUBLE PRECISION
         EVEN SPACING
 
           Order of data in file          RY1    IY1    RY2    IY2
                                          RY3    IY3    RY4    IY4
                                           .
                                           .
                                           .
 
           Input
 
                     DOUBLE PRECISION Y(2,2)
                       .
                       .
                       .
                     NPRO=0
                  10 READ(LUN,1000,ERR=  ,END=  )((Y(I,J),I=1,2),J=1,2)
                1000 FORMAT(4E20.12)
                     NPRO=NPRO+2
                       .
                       .    code to process these two values
                       .
                     IF(NPRO.LT.NVAL)GO TO 10
                       .
                       .   continued processing
                       .
 
           Output
 
                     DOUBLE PRECISION Y(2,2)
                       .
                       .
                       .
                     NPRO=0
                  10 CONTINUE
                       .
                       .    code to set up these two values
                       .
                     WRITE(LUN,1000,ERR=  )((Y(I,J),I=1,2),J=1,2)
                1000 FORMAT(4E20.12)
                     NPRO=NPRO+2
                     IF(NPRO.LT.NVAL)GO TO 10
                       .
                       .   continued processing
                       .
         CASE 8
 
         COMPLEX
         DOUBLE PRECISION
         UNEVEN SPACING
 
           Order of data in file          X1   RY1    IY1
                                          X2   RY2    IY2
                                           .
                                           .
                                           .
           Input
 
                     REAL X
                     DOUBLE PRECISION Y(2)
                       .
                       .
                       .
                     NPRO=0
                  10 READ(LUN,1000,ERR=  ,END=  )(X,Y(I),I=1,2)
                1000 FORMAT(E13.5,2E20.12)
                     NPRO=NPRO+1
                       .
                       .    code to process this value
                       .
                     IF(NPRO.LT.NVAL)GO TO 10
                       .
                       .   continued processing
                       .
 
           Output
 
                     REAL X
                     DOUBLE PRECISION Y(2)
                       .
                       .
                       .
                     NPRO=0
                  10 CONTINUE
                       .
                       .    code to set up this value
                       .
                     WRITE(LUN,1000,ERR=  )(X,Y(I),I=1,2)
                1000 FORMAT(E13.5,2E20.12)
                     NPRO=NPRO+1
                     IF(NPRO.LT.NVAL)GO TO 10
                       .
                       .   continued processing
                       .
 
-----------------------------------------------------------------------
The Binary 58 Universal File Format (UFF):

The basic (ASCII) universal file format for data is universal file format
58.  This format is completely documented by SDRC and a copy of that
documentation is on the UC-SDRL web site (58.asc). The
universal file format always begins with two records that are prior to the
information defined by each universal file format and ends with a record
that is placed after the information defined by the format.   First of
all, all records are 80 character ASCII records for the basic universal
file format. The first and last record are start/stop records and are
always -1 in the first six columns, right justified (Fortran I6 field
with -1 in the field).  The second record (Identifier Record) always
contains the universal file format number in the first 6 columns, right
justified.

This gives a file structure as follows (where b represent a blank
character):

bbbb-1
bbbb58
...
...
...
bbbb-1

The Binary 58 universal file format was originally developed by UC-SDRL 
in order to eliminate the need to compress the UFF 58 records and to reduce
the time required to load the UFF 58 data records.  The Binary 58 universal file
format yields files that are comparable to compressed files (approximately 3 to
4 times smaller than the equivalent UFF 58 file).  The Binary 58 universal file 
format loads approximately 30 to 40 times faster than the equivalent UFF 58 
file, depending upon the computing environment.  This new format was 
submitted to SDRC and subsequently adopted as a supported format.

The Binary 58 universal file format uses the same ASCII records at the
start of each data file as the ASCII dataset 58 but, beginning with
record 12, the data is stored in binary form rather than the specified
ASCII format.  The identifier record has the same 58 identifier in the
first six columns, right justified, but has additional information in
the rest of the 80 character record that identifies the binary format
(the size of the binary record, the format of the binary structure, etc.).

    -1
    58b     x     y          11        zzzz     0     0           0           0
...
... (11 ASCII header lines)
...
...
... (zzzz BINARY bytes of data, in format specifed by x and y, above)
... (interleaved as specified by the ASCII dataset 58)
...
    -1


When reading or writing a dataset 58b, care must be taken that the
binary data immediately follows the ASCII header lines and the closing
'    -1' immediately follows the binary data.  The binary data content
is written in the same sequence as the ASCII dataset 58 (ie. field
order sequence).  The field size is NOT used, however the data type
(int/float/double) content is.  Note: there are no CR/LF characters
embedded in or following the binary data.


=====================================================================
The Format of 58b ID-Line:
----------------------------

For the traditional dataset 58 (Function at Nodal DOF), the dataset
id-line is composed of four spaces followed by "58". This line has been
enhanced to contain additional information for the binary version of
dataset 58.

    -1
    58b     2     2          11        4096     0     0           0           0

     Format (I6,1A1,I6,I6,I12,I12,I6,I6,I12,I12)

              Field 1       - 58  [I6]
              Field 2       - lowercase b [1A1]
              Field 3       - Byte Ordering Method [I6]
                              1 - Little Endian (DEC VMS & ULTRIX, WIN NT)
                              2 - Big Endian (most UNIX)
              Field 4       - Floating Point Format [I6]
                              1 - DEC VMS
                              2 - IEEE 754 (UNIX)
                              3 - IBM 5/370
              Field 5       - number of ASCII lines following  [I12]
                              11 - for dataset 58
              Field 6       - number of bytes following ASCII lines  [I12]
              Fields 7-10   - Not used (fill with zeros)


The format of this line should remain constant for any other dataset
that takes on a binary format in the future.

=====================================================================
"""
    if raw:
        return out
    else:
        print(out)   

def _write58(fh, dset, mode='add', _filename=None, force_double=True):
    """Writes function at nodal DOF - data-set 58 - to an open file fh."""
    try:
        if not (dset['func_type'] in [0, 1, 2, 3, 4, 6, 9]):
            raise ValueError('Unsupported function type')
        # handle optional fields - only those that are not calculated
        # automatically
        dict = {'units_description': '',
                'id1': 'NONE',
                'id2': 'NONE',
                'id3': 'NONE',
                'id4': 'NONE',
                'id5': 'NONE',
                'func_id': 0,
                'ver_num': 0,
                'binary': 0,
                'load_case_id': 0,
                'rsp_ent_name': 'NONE',
                'ref_ent_name': 'NONE',
                'abscissa_axis_lab': 'NONE',
                'abscissa_axis_units_lab': 'NONE',
                'abscissa_len_unit_exp': 0,
                'abscissa_force_unit_exp': 0,
                'abscissa_temp_unit_exp': 0,
                'ordinate_len_unit_exp': 0,
                'ordinate_force_unit_exp': 0,
                'ordinate_temp_unit_exp': 0,
                'ordinate_axis_lab': 'NONE',
                'ordinate_axis_units_lab': 'NONE',
                'orddenom_len_unit_exp': 0,
                'orddenom_force_unit_exp': 0,
                'orddenom_temp_unit_exp': 0,
                'orddenom_axis_lab': 'NONE',
                'orddenom_axis_units_lab': 'NONE',
                'z_axis_len_unit_exp': 0,
                'z_axis_force_unit_exp': 0,
                'z_axis_temp_unit_exp': 0,
                'z_axis_axis_lab': 'NONE',
                'z_axis_axis_units_lab': 'NONE',
                'z_axis_value': 0,
                'spec_data_type': 0,
                'abscissa_spec_data_type': 0,
                'ordinate_spec_data_type': 0,
                'z_axis_spec_data_type': 0,
                'version_num': 0,
                'abscissa_spacing': 0}
        dset = _opt_fields(dset, dict)

        # Write strings to the file - always in double precision 
        # => ord_data_type = 2 (single) and 4 (double) for real data 
        # => ord_data_type = 5 (single) and 6 (double) for complex data
        num_pts = len(dset['data'])
        is_r = not np.iscomplexobj(dset['data'])
        if is_r:
            # real data
            if force_double:
                dset['ord_data_type'] = 4
        else:
            # complex data
            if force_double:
                dset['ord_data_type'] = 6

        if dset['ord_data_type'] in [4, 6]: # double precision
            is_double = True
            n_bytes = num_pts * 8
        elif dset['ord_data_type'] in [2, 5]: # sigle precision
            is_double = False
            n_bytes = num_pts * 4

        if 'n_bytes' in dset.keys():
            dset['n_bytes'] = n_bytes

        ord_data_type = dset['ord_data_type']

        is_even = bool(dset['abscissa_spacing'])  # handling even/uneven abscissa spacing manually

        # handling abscissa spacing automatically
        # is_even = len( set( [ dset['x'][ii]-dset['x'][ii-1] for ii in range(1,len(dset['x'])) ] ) ) == 1
        # decode utf to ascii
        for k, v in dset.items():
            if type(v) == str:
                dset[k] = v.encode("utf-8").decode('ascii','ignore')

        dset['abscissa_min'] = dset['x'][0]
        dx = dset['x'][1] - dset['x'][0]
        fh.write('%6i\n%6i' % (-1, 58))
        if dset['binary']:
            if sys.byteorder == 'little':
                bo = 1
            else:
                bo = 2
            fh.write('b%6i%6i%12i%12i%6i%6i%12i%12i\n' % (bo, 2, 11, n_bytes, 0, 0, 0, 0))
        else:
            fh.write('%74s\n' % ' ')
        fh.write('%-80s\n' % dset['id1'])
        fh.write('%-80s\n' % dset['id2'])
        fh.write('%-80s\n' % dset['id3'])
        fh.write('%-80s\n' % dset['id4'])
        fh.write('%-80s\n' % dset['id5'])
        fh.write('%5i%10i%5i%10i %10s%10i%4i %10s%10i%4i\n' %
                    (dset['func_type'], dset['func_id'], dset['ver_num'], dset['load_case_id'],
                    dset['rsp_ent_name'], dset['rsp_node'], dset['rsp_dir'], dset['ref_ent_name'],
                    dset['ref_node'], dset['ref_dir']))
        fh.write('%10i%10i%10i%13.5e%13.5e%13.5e\n' % (ord_data_type, num_pts, is_even,
                                                        is_even * dset['abscissa_min'], is_even * dx,
                                                        dset['z_axis_value']))
        fh.write('%10i%5i%5i%5i %-20s %-20s\n' % (dset['abscissa_spec_data_type'],
                                                    dset['abscissa_len_unit_exp'], dset['abscissa_force_unit_exp'],
                                                    dset['abscissa_temp_unit_exp'], dset['abscissa_axis_lab'],
                                                    dset['abscissa_axis_units_lab']))
        fh.write('%10i%5i%5i%5i %-20s %-20s\n' % (dset['ordinate_spec_data_type'],
                                                    dset['ordinate_len_unit_exp'], dset['ordinate_force_unit_exp'],
                                                    dset['ordinate_temp_unit_exp'], dset['ordinate_axis_lab'],
                                                    dset['ordinate_axis_units_lab']))
        fh.write('%10i%5i%5i%5i %-20s %-20s\n' % (dset['orddenom_spec_data_type'],
                                                    dset['orddenom_len_unit_exp'], dset['orddenom_force_unit_exp'],
                                                    dset['orddenom_temp_unit_exp'], dset['orddenom_axis_lab'],
                                                    dset['orddenom_axis_units_lab']))
        fh.write('%10i%5i%5i%5i %-20s %-20s\n' % (dset['z_axis_spec_data_type'],
                                                    dset['z_axis_len_unit_exp'], dset['z_axis_force_unit_exp'],
                                                    dset['z_axis_temp_unit_exp'], dset['z_axis_axis_lab'],
                                                    dset['z_axis_axis_units_lab']))
        if is_r:
            if is_even:
                data = dset['data'].copy()
            else:
                data = np.zeros(2 * num_pts, 'd')
                data[0:-1:2] = dset['x']
                data[1::2] = dset['data']
        else:
            if is_even:
                data = np.zeros(2 * num_pts, 'd')
                data[0:-1:2] = dset['data'].real
                data[1::2] = dset['data'].imag
            else:
                data = np.zeros(3 * num_pts, 'd')
                data[0:-2:3] = dset['x']
                data[1:-1:3] = dset['data'].real
                data[2::3] = dset['data'].imag
        # always write data in double precision
        if dset['binary']:
            fh.close()
            if mode.lower() == 'overwrite':
                fh = open(_filename, 'wb')
            elif mode.lower() == 'add':
                fh = open(_filename, 'ab')
            # write data
            if bo == 1:
                [fh.write(struct.pack('<d', datai)) for datai in data]
            else:
                [fh.write(struct.pack('>d', datai)) for datai in data]
            fh.close()
            if mode.lower() == 'overwrite':
                fh = open(_filename, 'wt')
            elif mode.lower() == 'add':
                fh = open(_filename, 'at')
        else:
            if is_double: # write double precision
                n4_blocks = len(data) // 4
                rem_vals = len(data) % 4
                if is_r:
                    if is_even:
                        fh.write(n4_blocks * '%20.11e%20.11e%20.11e%20.11e\n' % tuple(data[:4 * n4_blocks]))
                        if rem_vals > 0:
                            fh.write((rem_vals * '%20.11e' + '\n') % tuple(data[4 * n4_blocks:]))
                    else:
                        fh.write(n4_blocks * '%13.5e%20.11e%13.5e%20.11e\n' % tuple(data[:4 * n4_blocks]))
                        if rem_vals > 0:
                            fmt = ['%13.5e', '%20.11e', '%13.5e', '%20.11e']
                            fh.write((''.join(fmt[rem_vals]) + '\n') % tuple(data[4 * n4_blocks:]))
                else:
                    if is_even:
                        fh.write(n4_blocks * '%20.11e%20.11e%20.11e%20.11e\n' % tuple(data[:4 * n4_blocks]))
                        if rem_vals > 0:
                            fh.write((rem_vals * '%20.11e' + '\n') % tuple(data[4 * n4_blocks:]))
                    else:
                        n3_blocks = len(data) // 3
                        rem_vals = len(data) % 3
                        fh.write(n3_blocks * '%13.5e%20.11e%20.11e\n' % tuple(data[:3 * n3_blocks]))
                        if rem_vals != 0: # There should be no rem
                            print('Warning: Something went wrong when savning the uff file.')

            else: # single precision
                n6_blocks = len(data) // 6
                rem_vals = len(data) % 6
                fh.write(n6_blocks * '%13.5e%13.5e%13.5e%13.5e%13.5e%13.5e\n' % tuple(data[:6 * n6_blocks]))
                if rem_vals > 0:
                    fh.write((rem_vals * '%13.5e' + '\n') % tuple(data[6 * n6_blocks:]))

        fh.write('%6i\n' % -1)
        del data
    except KeyError as msg:
        raise Exception('The required key \'' + msg.args[0] + '\' not present when writing data-set #58')
    except:
        raise Exception('Error writing data-set #58')


def _extract58(block_data, header_only=False):
    """
    Extract function at nodal DOF - data-set 58. 

    :param header_only: False (default). If True the header data will be 
                        extracted, only (useful with large files).
    """





    dset = {'type': 58, 'binary': 0}
    try:
        binary = False
        split_header = b''.join(block_data.splitlines(True)[:13]).decode('utf-8',  errors='replace').splitlines(True)
        if len(split_header[1]) >= 7:
            if split_header[1][6].lower() == 'b':
                # Read some addititional fields from the header section
                binary = True
                dset['binary'] = 1
                dset.update(_parse_header_line(split_header[1], 6, [6, 1, 6, 6, 12, 12, 6, 6, 12, 12],
                                                    [-1, -1, 2, 2, 2, 2, -1, -1, -1, -1],
                                                    ['', '', 'byte_ordering', 'fp_format', 'n_ascii_lines',
                                                        'n_bytes', '', '', '', '']))
        dset.update(_parse_header_line(split_header[2], 1, [80], [1], ['id1']))
        dset.update(_parse_header_line(split_header[3], 1, [80], [1], ['id2']))
        dset.update(_parse_header_line(split_header[4], 1, [80], [1], ['id3']))  # usually for the date
        dset.update(_parse_header_line(split_header[5], 1, [80], [1], ['id4']))
        dset.update(_parse_header_line(split_header[6], 1, [80], [1], ['id5']))
        dset.update(_parse_header_line(split_header[7], 1, [5, 10, 5, 10, 11, 10, 4, 11, 10, 4],
                                            [2, 2, 2, 2, 1, 2, 2, 1, 2, 2],
                                            ['func_type', 'func_id', 'ver_num', 'load_case_id', 'rsp_ent_name',
                                                'rsp_node', 'rsp_dir', 'ref_ent_name',
                                                'ref_node', 'ref_dir']))
        dset.update(_parse_header_line(split_header[8], 6, [10, 10, 10, 13, 13, 13], [2, 2, 2, 3, 3, 3],
                                            ['ord_data_type', 'num_pts', 'abscissa_spacing', 'abscissa_min',
                                                'abscissa_inc', 'z_axis_value']))
        dset.update(_parse_header_line(split_header[9], 4, [10, 5, 5, 5, 21, 21], [2, 2, 2, 2, 1, 1],
                                            ['abscissa_spec_data_type', 'abscissa_len_unit_exp',
                                                'abscissa_force_unit_exp', 'abscissa_temp_unit_exp',
                                                'abscissa_axis_lab', 'abscissa_axis_units_lab']))
        dset.update(_parse_header_line(split_header[10], 4, [10, 5, 5, 5, 21, 21], [2, 2, 2, 2, 1, 1],
                                            ['ordinate_spec_data_type', 'ordinate_len_unit_exp',
                                                'ordinate_force_unit_exp', 'ordinate_temp_unit_exp',
                                                'ordinate_axis_lab', 'ordinate_axis_units_lab']))
        dset.update(_parse_header_line(split_header[11], 4, [10, 5, 5, 5, 21, 21], [2, 2, 2, 2, 1, 1],
                                            ['orddenom_spec_data_type', 'orddenom_len_unit_exp',
                                                'orddenom_force_unit_exp', 'orddenom_temp_unit_exp',
                                                'orddenom_axis_lab', 'orddenom_axis_units_lab']))
        dset.update(_parse_header_line(split_header[12], 4, [10, 5, 5, 5, 21, 21], [2, 2, 2, 2, 1, 1],
                                            ['z_axis_spec_data_type', 'z_axis_len_unit_exp',
                                                'z_axis_force_unit_exp', 'z_axis_temp_unit_exp', 'z_axis_axis_lab',
                                                'z_axis_axis_units_lab']))
        # Body
        # split_data = ''.join(split_data[13:])
        if header_only:
            # If not reading data, just set placeholders
            dset['x'] = None
            dset['data'] = None
        else:
            if binary:
                try:     
                    split_data = b''.join(block_data.splitlines(True)[13:])
                    if dset['byte_ordering'] == 1:
                        bo = '<'
                    else:
                        bo = '>'
                    if (dset['ord_data_type'] == 2) or (dset['ord_data_type'] == 5):
                        # single precision - 4 bytes
                        values = np.asarray(struct.unpack('%c%sf' % (bo, int(len(split_data) / 4)), split_data), 'd')
                    else:
                        # double precision - 8 bytes
                        values = np.asarray(struct.unpack('%c%sd' % (bo, int(len(split_data) / 8)), split_data), 'd')
                except:
                    raise Exception('Potentially wrong data format (common with binary files from some commercial softwares). Try using pyuff.fix_58b() to fix your file. For more information, see https://github.com/ladisk/pyuff/issues/61')
            else:
                values = []
                split_data = block_data.decode('utf-8', errors='replace').splitlines(True)[13:]
                if (dset['ord_data_type'] == 2) or (dset['ord_data_type'] == 5):
                    for line in split_data[:-1]:  # '6E13.5'
                        values.extend([float(line[13 * i:13 * (i + 1)]) for i in range(len(line) // 13)])
                    else:
                        line = split_data[-1]
                        values.extend([float(line[13 * i:13 * (i + 1)]) for i in range(len(line) // 13) if line[13 * i:13 * (i + 1)]!='             '])
                elif ((dset['ord_data_type'] == 4) or (dset['ord_data_type'] == 6)) and (dset['abscissa_spacing'] == 1):
                    for line in split_data:  # '4E20.12'
                        values.extend([float(line[20 * i:20 * (i + 1)]) for i in range(len(line) // 20)])
                elif (dset['ord_data_type'] == 4) and (dset['abscissa_spacing'] == 0):
                    for line in split_data:  # 2(E13.5,E20.12)
                        values.extend(
                            [float(line[13 * (i + j) + 20 * (i):13 * (i + 1) + 20 * (i + j)]) \
                                for i in range(len(line) // 33) for j in [0, 1]])
                elif (dset['ord_data_type'] == 6) and (dset['abscissa_spacing'] == 0):
                    for line in split_data:  # 1E13.5,2E20.12
                        values.extend([float(line[0:13]), float(line[13:33]), float(line[33:53])])
                else:
                    raise Exception('Error reading data-set #58b; not proper data case.')

                values = np.asarray(values)
                # values = np.asarray([float(str) for str in split_data],'d')
            if (dset['ord_data_type'] == 2) or (dset['ord_data_type'] == 4):
                # Non-complex ordinate data
                if (dset['abscissa_spacing'] == 0):
                    # Uneven abscissa
                    dset['x'] = values[:-1:2].copy()
                    dset['data'] = values[1::2].copy()
                else:
                    # Even abscissa
                    n_val = len(values)
                    min_val = dset['abscissa_min']
                    d = dset['abscissa_inc']
                    dset['x'] = min_val + np.arange(n_val) * d
                    dset['data'] = values.copy()
            elif (dset['ord_data_type'] == 5) or (dset['ord_data_type'] == 6):
                # Complex ordinate data
                if (dset['abscissa_spacing'] == 0):
                    # Uneven abscissa
                    dset['x'] = values[:-2:3].copy()
                    dset['data'] = values[1:-1:3] + 1.j * values[2::3]
                else:
                    # Even abscissa
                    n_val = len(values) / 2
                    min_val = dset['abscissa_min']
                    d = dset['abscissa_inc']
                    dset['x'] = min_val + np.arange(n_val) * d
                    dset['data'] = values[0:-1:2] + 1.j * values[1::2]
            del values
    except:
        raise Exception('Error reading data-set #58b')
    return dset


def prepare_58(
        binary=None,
        id1=None,
        id2=None,
        id3=None,
        id4=None,
        id5=None,

        func_type=None,
        ver_num=None,
        load_case_id=None,
        rsp_ent_name=None,
        rsp_node=None,
        rsp_dir=None,
        ref_ent_name=None,
        ref_node=None,
        ref_dir=None,

        ord_data_type=None,
        num_pts=None,
        abscissa_spacing=None,
        abscissa_min=None,
        abscissa_inc=None,
        z_axis_value=None,

        abscissa_spec_data_type=None,
        abscissa_len_unit_exp=None,
        abscissa_force_unit_exp=None,
        abscissa_temp_unit_exp=None,
        
        abscissa_axis_units_lab=None,

        ordinate_spec_data_type=None,
        ordinate_len_unit_exp=None,
        ordinate_force_unit_exp=None,
        ordinate_temp_unit_exp=None,
        
        ordinate_axis_units_lab=None,

        orddenom_spec_data_type=None,
        orddenom_len_unit_exp=None,
        orddenom_force_unit_exp=None,
        orddenom_temp_unit_exp=None,
        
        orddenom_axis_units_lab=None,

        z_axis_spec_data_type=None,
        z_axis_len_unit_exp=None,
        z_axis_force_unit_exp=None,
        z_axis_temp_unit_exp=None,
        
        z_axis_axis_units_lab=None,

        data=None,
        x=None,
        spec_data_type=None,
        byte_ordering=None,
        fp_format=None,
        n_ascii_lines=None,
        n_bytes=None,
        return_full_dict=False):

    """Name:   Function at Nodal DOF

    R-Record, F-Field

    :param binary: 1 for binary, 0 for ascii, optional
    :param id1: R1 F1, ID Line 1, optional
    :param id2: R2 F1, ID Line 2, optional
    :param id3: R3 F1, ID Line 3, optional
    :param id4: R4 F1, ID Line 4, optional
    :param id5: R5 F1, ID Line 5, optional

    **DOF identification**

    :param func_type: R6 F1, Function type
    :param ver_num: R6 F3, Version number, optional
    :param load_case_id: R6 F4, Load case identification number, optional
    :param rsp_ent_name: R6 F5, Response entity name, optional
    :param rsp_node: R6 F6, Response node
    :param rsp_dir: R6 F7, Responde direction
    :param ref_ent_name: R6 F8, Reference entity name, optional
    :param ref_node: R6 F9, Reference node
    :param ref_dir: R6 F10, Reference direction

    **Data form**

    :param ord_data_type: R7 F1, Ordinate data type, ignored
    :param num_pts: R7 F2, number of data pairs for uneven abscissa or number of data values for even abscissa, ignored
    :param abscissa_spacing: R7 F3, Abscissa spacing (0- uneven, 1-even), ignored
    :param abscissa_min: R7 F4, Abscissa minimum (0.0 if spacing uneven), ignored
    :param abscissa_inc: R7 F5, Abscissa increment (0.0 if spacing uneven), ignored
    :param z_axis_value: R7 F6, Z-axis value (0.0 if unused), optional

    **Abscissa data characteristics**

    :param abscissa_spec_data_type: R8 F1, Abscissa specific data type, optional
    :param abscissa_len_unit_exp: R8 F2, Abscissa length units exponent, optional
    :param abscissa_force_unit_exp: R8 F3, Abscissa force units exponent, optional
    :param abscissa_temp_unit_exp: R8 F4, Abscissa temperature units exponent, optional
    
    :param abscissa_axis_units_lab: R8 F6, Abscissa units label, optional

    **Ordinate (or ordinate numerator) data characteristics**

    :param ordinate_spec_data_type: R9 F1, Ordinate specific data type, optional
    :param ordinate_len_unit_exp: R9 F2, Ordinate length units exponent, optional
    :param ordinate_force_unit_exp: R9 F3, Ordinate force units exponent, optional
    :param ordinate_temp_unit_exp: R9 F4, Ordinate temperature units exponent, optional
    
    :param ordinate_axis_units_lab: R9 F6, Ordinate units label, optional

    **Ordinate denominator data characteristics**

    :param orddenom_spec_data_type: R10 F1, Ordinate Denominator specific data type, optional
    :param orddenom_len_unit_exp: R10 F2, Ordinate Denominator length units exponent, optional
    :param orddenom_force_unit_exp: R10 F3, Ordinate Denominator force units exponent, optional
    :param orddenom_temp_unit_exp: R10 F4, Ordinate Denominator temperature units exponent, optional
    
    :param orddenom_axis_units_lab: R10 F6, Ordinate Denominator units label, optional

    **Z-axis data characteristics**

    :param z_axis_spec_data_type:  R11 F1, Z-axis specific data type, optional
    :param z_axis_len_unit_exp: R11 F2, Z-axis length units exponent, optional
    :param z_axis_force_unit_exp: R11 F3, Z-axis force units exponent, optional
    :param z_axis_temp_unit_exp: R11 F4, Z-axis temperature units exponent, optional
    
    :param z_axis_axis_units_lab: R11 F6, Z-axis units label, optional

    **Data values**

    :param data: R12 F1, Data values

    :param x: Abscissa array
    :param spec_data_type: Specific data type, optional
    :param byte_ordering: R1 F3, Byte ordering (only for binary), ignored
    :param fp_format: R1 F4 Floating-point format (only for binary), ignored
    :param n_ascii_lines: R1 F5, Number of ascii lines (only for binary), ignored
    :param n_bytes: R1 F6, Number of bytes (only for binary), ignored

    :param return_full_dict: If True full dict with all keys is returned, else only specified arguments are included

    **Test prepare_58**

    >>> save_to_file = 'test_pyuff'
    >>> if save_to_file:
    >>>     if os.path.exists(save_to_file):
    >>>         os.remove(save_to_file)
    >>> uff_datasets = []
    >>> binary = [0, 1, 0]  # ascii of binary
    >>> frequency = np.arange(10)
    >>> np.random.seed(0)
    >>> for i, b in enumerate(binary):
    >>>     print('Adding point {}'.format(i + 1))
    >>>     response_node = 1
    >>>     response_direction = 1
    >>>     reference_node = i + 1
    >>>     reference_direction = 1
    >>>     # this is an artificial 'frf'
    >>>     acceleration_complex = np.random.normal(size=len(frequency)) + 1j * np.random.normal(size=len(frequency))
    >>>     name = 'TestCase'
    >>>     data = pyuff.prepare_58(
    >>>         binary=binary[i],
    >>>         func_type=4,
    >>>         rsp_node=response_node,
    >>>         rsp_dir=response_direction,
    >>>         ref_dir=reference_direction,
    >>>         ref_node=reference_node,
    >>>         data=acceleration_complex,
    >>>         x=frequency,
    >>>         id1='id1',
    >>>         rsp_ent_name=name,
    >>>         ref_ent_name=name,
    >>>         abscissa_spacing=1,
    >>>         abscissa_spec_data_type=18,
    >>>         ordinate_spec_data_type=12,
    >>>         orddenom_spec_data_type=13)
    >>>     uff_datasets.append(data.copy())
    >>>     if save_to_file:
    >>>         uffwrite = pyuff.UFF(save_to_file)
    >>>         uffwrite._write_set(data, 'add')
    >>> uff_datasets
    """

    if binary not in (0, 1, None):
        raise ValueError('binary can be 0 or 1')
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
    
    if func_type not in np.arange(28) and func_type != None:
        raise ValueError('func_type must be integer between 0 and 27')
    if np.array(ver_num).dtype != int and ver_num != None:
        raise TypeError('ver_num must be integer')
    if np.array(load_case_id).dtype != int and load_case_id != None:
        raise TypeError('load_case_id must be integer')
    if type(rsp_ent_name) != str and rsp_ent_name != None:
        raise TypeError('rsp_ent_name must be string')
    if np.array(rsp_node).dtype != int and rsp_node != None:
        raise TypeError('rsp_node must be integer')
    if rsp_dir not in np.arange(-6,7) and rsp_dir != None:
        raise ValueError('rsp_dir must be integer between -6 and 6')
    if type(ref_ent_name) != str and ref_ent_name != None:
        raise TypeError('rsp_ent_name must be string')
    if np.array(ref_node).dtype != int and ref_node != None:
        raise TypeError('ref_node must be int')
    if ref_dir not in np.arange(-6,7) and ref_dir != None:
        raise ValueError('ref_dir must be integer between -6 and 6')
    
    if ord_data_type not in (2, 4, 5, 6, None):
        raise ValueError('ord_data_type can be: 2,4,5,6')
    if np.array(num_pts).dtype != int and num_pts != None:
        raise TypeError('num_pts must be integer')
    if abscissa_spacing not in (0, 1, None):
        raise ValueError('abscissa_spacing can be 0:uneven, 1:even')
    if np.array(abscissa_min).dtype != float and abscissa_min != None:
        raise TypeError('abscissa_min must be float')
    if np.array(abscissa_inc).dtype != float and abscissa_inc != None:
        raise TypeError('abscissa_inc must be float')
    if np.array(z_axis_value).dtype != float and z_axis_value != None:
        raise TypeError('z_axis_value must be float')
    
    if abscissa_spec_data_type not in np.arange(21) and abscissa_spec_data_type != None:
        raise ValueError('abscissa_spec_data_type must be integer between 0 nd 21')
    if np.array(abscissa_len_unit_exp).dtype != int and abscissa_len_unit_exp != None:
        raise TypeError('abscissa_len_unit_exp must be integer')
    if np.array(abscissa_force_unit_exp).dtype != int and abscissa_force_unit_exp != None:
        raise TypeError('abscissa_force_unit_exp must be integer')
    if np.array(abscissa_temp_unit_exp).dtype != int and abscissa_temp_unit_exp != None:
        raise TypeError('abscissa_temp_unit_exp must be integer')
    if type(abscissa_axis_units_lab) != str and abscissa_axis_units_lab != None:
        raise TypeError('abscissa_axis_units_lab must be string')

    if ordinate_spec_data_type not in np.arange(21) and ordinate_spec_data_type != None:
        raise ValueError('ordinate_spec_data_type must be integer between 0 nd 21')
    if np.array(ordinate_len_unit_exp).dtype != int and ordinate_len_unit_exp != None:
        raise TypeError('ordinate_len_unit_exp must be integer')
    if np.array(ordinate_force_unit_exp).dtype != int and ordinate_force_unit_exp != None:
        raise TypeError('ordinate_force_unit_exp must be integer')
    if np.array(ordinate_temp_unit_exp).dtype != int and ordinate_temp_unit_exp != None:
        raise TypeError('ordinate_temp_unit_exp must be integer')
    if type(ordinate_axis_units_lab) != str and ordinate_axis_units_lab != None:
        raise TypeError('ordinate_axis_units_lab must be string')

    if orddenom_spec_data_type not in np.arange(21) and orddenom_spec_data_type != None:
        raise ValueError('orddenom_spec_data_type must be integer between 0 nd 21')
    if np.array(orddenom_len_unit_exp).dtype != int and orddenom_len_unit_exp != None:
        raise TypeError('orddenom_len_unit_exp must be integer')
    if np.array(orddenom_force_unit_exp).dtype != int and orddenom_force_unit_exp != None:
        raise TypeError('orddenom_force_unit_exp must be integer')
    if np.array(orddenom_temp_unit_exp).dtype != int and orddenom_temp_unit_exp != None:
        raise TypeError('orddenom_temp_unit_exp must be integer')
    if type(orddenom_axis_units_lab) != str and orddenom_axis_units_lab != None:
        raise TypeError('orddenom_axis_units_lab must be string')

    if z_axis_spec_data_type not in np.arange(21) and z_axis_spec_data_type != None:
        raise ValueError('z_axis_spec_data_type must be integer between 0 nd 21')
    if np.array(z_axis_len_unit_exp).dtype != int and z_axis_len_unit_exp != None:
        raise TypeError('z_axis_len_unit_exp must be integer')
    if np.array(z_axis_force_unit_exp).dtype != int and z_axis_force_unit_exp != None:
        raise TypeError('z_axis_force_unit_exp must be integer')
    if np.array(z_axis_temp_unit_exp).dtype != int and z_axis_temp_unit_exp != None:
        raise TypeError('z_axis_temp_unit_exp must be integer')
    if type(z_axis_axis_units_lab) != str and z_axis_axis_units_lab != None:
        raise TypeError('z_axis_axis_units_lab must be string')
    
    if np.array(data).dtype != float and np.array(data).dtype != complex:
        if data != None:
            raise TypeError('data must be float')
    


    dataset={
        'type': 58,
        'binary': binary,
        'id1': id1,
        'id2': id2,
        'id3': id3,
        'id4': id4,
        'id5': id5,

        'func_type': func_type,
        'ver_num': ver_num,
        'load_case_id': load_case_id,
        'rsp_ent_name': rsp_ent_name,
        'rsp_node': rsp_node,
        'rsp_dir': rsp_dir,
        'ref_ent_name': ref_ent_name,
        'ref_node': ref_node,
        'ref_dir': ref_dir,

        'ord_data_type': ord_data_type,
        'num_pts': num_pts,
        'abscissa_spacing': abscissa_spacing,
        'abscissa_min': abscissa_min,
        'abscissa_inc': abscissa_inc,
        'z_axis_value': z_axis_value,

        'abscissa_spec_data_type': abscissa_spec_data_type,
        'abscissa_len_unit_exp': abscissa_len_unit_exp,
        'abscissa_force_unit_exp': abscissa_force_unit_exp,
        'abscissa_temp_unit_exp': abscissa_temp_unit_exp,
        
        'abscissa_axis_units_lab': abscissa_axis_units_lab,

        'ordinate_spec_data_type': ordinate_spec_data_type,
        'ordinate_len_unit_exp': ordinate_len_unit_exp,
        'ordinate_force_unit_exp': ordinate_force_unit_exp,
        'ordinate_temp_unit_exp': ordinate_temp_unit_exp,
        
        'ordinate_axis_units_lab': ordinate_axis_units_lab,

        'orddenom_spec_data_type': orddenom_spec_data_type,
        'orddenom_len_unit_exp': orddenom_len_unit_exp,
        'orddenom_force_unit_exp': orddenom_force_unit_exp,
        'orddenom_temp_unit_exp': orddenom_temp_unit_exp,
        
        'orddenom_axis_units_lab': orddenom_axis_units_lab,

        'z_axis_spec_data_type': z_axis_spec_data_type,
        'z_axis_len_unit_exp': z_axis_len_unit_exp,
        'z_axis_force_unit_exp': z_axis_force_unit_exp,
        'z_axis_temp_unit_exp': z_axis_temp_unit_exp,
        
        'z_axis_axis_units_lab': z_axis_axis_units_lab,

        'data': data,
        'x': x,
        'spec_data_type': spec_data_type,
        'byte_ordering': byte_ordering,
        'fp_format': fp_format,
        'n_ascii_lines': n_ascii_lines,
        'n_bytes': n_bytes
        }

    if return_full_dict is False:
        dataset = check_dict_for_none(dataset)


    return dataset


def fix_58b(filename,fixed_filename=None):
    """
    Opens the UFF file, fixes a common formatting issue and saves the fixed file. 
    Specifically, it fixes the instance, when closing '    -1' of the dataset is on its own line, and not right after the data.

    :param filename: filename of the UFF file to be fixed
    :param filename: filename to write the fixed UFF file, if None, the fixed file will be saved as 'filename_fixed.uff'
    """
    
    if not os.path.exists(filename):
        raise Exception('Filename does not exist')
    try:
        # Open the file in binary read mode
        with open(filename, 'rb') as fh:
            data = fh.read()
    except Exception as e:
        raise Exception(f'Cannot access the file {filename}: {e}')
    else:
        try:
            lines = data.splitlines(keepends=True)

            # Fix 1: Adjust ending '    -1' line
            if len(lines) >= 1 and lines[-1].strip() == b'-1':
                if len(lines) >= 2:
                    # Move '    -1' up to the end of the previous line
                    prev_line = lines[-2].rstrip(b'\r\n')
                    prev_line += b'    -1' + lines[-1][-1:]  # Keep the newline character
                    lines[-2] = prev_line
                    lines.pop()  # Remove the last line
                else:
                    pass

            # Fix 2: Adjust 'data\n    -1\n    -1\n data' patterns
            i = 0
            while i < len(lines) - 3:
                if (lines[i+1].strip() == b'-1' and lines[i+2].strip() == b'-1'):
                    # Move '    -1' from lines[i+1] to the end of lines[i]
                    data_line = lines[i].rstrip(b'\r\n')  # Remove newline characters
                    data_line += b'    -1' + lines[i+1][-1:]  # Add '    -1' and newline
                    lines[i] = data_line
                    del lines[i+1]  # Remove the now-empty line
                    # Do not increment i to recheck the new line at position i
                else:
                    i += 1  # Move to the next line

            # Reassemble the data
            data = b''.join(lines)


            # Write the fixed data back to the file
            if fixed_filename is None:
                base, ext = os.path.splitext(filename)
                new_filename = f"{base}_fixed{ext}" #default filename
            else:
                new_filename = fixed_filename #custom filename
            with open(new_filename, 'wb') as fh:
                fh.write(data)
            print('fixed file saved as:', new_filename)
        except Exception as e:
            raise Exception(f'Error fixing UFF file: {filename}: {e}')

