#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2014-2017 Primož Čermelj, Matjaž Mršnik, Miha Pirnat, Janko Slavič, Blaž Starc (in alphabetic order)
# 
# This file is part of pyuff.
# 
# pyFRF is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License.
# 
# pyuff is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with pyuff.  If not, see <http://www.gnu.org/licenses/>.
"""
==========
pyuff module
==========

This module is part of the www.openmodal.com project and
defines an UFF class to manipulate with the
UFF (Universal File Format) files, i.e., to read from and write
to UFF files. Among the variety of UFF formats, only some of the
formats (data-set types) frequently used in structural dynamics
are supported: **151, 15, 55, 58, 58b, 82, 164, 2412, 2420.** Data-set **58b**
is actually a hybrid format [1]_ where the signal is written in the
binary form, while the header-part is slightly different from 58 but still in the
ascii format.

An UFF file is a file that can have many data-sets of either ascii or binary
data where data-set is a block of data between the start and end tags ``____-1``
(``_`` representing the space character). Refer to [1]_ and [2]_ for
more information about the UFF format.

This module also provides an exception handler class, ``UFFException``.

Sources:
    .. [1] https://www.ceas3.uc.edu/sdrluff/
    .. [2] Matlab's ``readuff`` and ``writeuff`` functions:
       http://www.mathworks.com/matlabcentral/fileexchange/loadFile.do?objectId=6395

Acknowledgement:
    * This source (py2.7) was first written in 2007, 2008 by Primoz Cermelj (primoz.cermelj@gmail.com)
    * As part of the www.openmodal.com project the first source was adopted for Python 3 by
      Matjaz Mrsnik  <matjaz.mrsnik@gmail.com>
    * The package is maintained by Janko Slavič <janko.slavic@fs.uni-lj.si>

Notes:
    * 58 data-set is always written in double precision, even if it is
      read in single precision.
    
    * ``numpy`` module is required as all the vector/matrix-type data are read
      or written using ``numpy.array`` objects.
      
    Example:
    >>> import pyuff
    >>> uff_file = pyuff.UFF('beam.uff')
    >>> uff_file.file_exists()
    True
"""
import os
import struct
import sys
import time

import numpy as np

__version__ = '1.25'
_SUPPORTED_SETS = ['151', '15', '55', '58', '58b', '82', '164', '2411', '2412', '2420','2414']


class UFFException(Exception):
    """An exception that prints a string describing the error.
    """

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return self.value


class UFF:
    """
    Manages data reading and writing from/to the UFF file.
    
    The UFF class instance requires exactly 1 parameter - a file name of a
    universal file. If the file does not exist, no basic file info will be
    extracted and the status will be False - indicating that the file is not
    refreshed. Hovewer, when one tries to read one or more data-sets, the file
    must exist or the UFFException will be raised.
    
    The file, given as a parameter to the UFF instance, is open only when
    reading from or writing to the file. The UFF instance refreshes the file
    automatically - use ``UFF.get_status()`` to see the refresh status); note
    that this works fine if the file is being changed only through the UFF
    instance and not by other functions or even by other means, e.g.,
    externally. If the file is changed externally, the ``UFF.refresh()`` should
    be invoked before any reading or writing.
    
    All array-type data are read/written using numpy's ``np.array`` module.
    
    Appendix
    --------
    Below are the fileds for all the data sets supported. <..> designates an
    *optional* field, i.e., a field that is not needed when writting to a file
    as the field has a defualt value. Additionally, <<..>> designates fields that
    are *ignored* when writing (not needed at all); these fields are defined
    automatically.
    
    Moreover, there are also some fields that are data-type dependent, i.e.,
    some fields that are onyl used/available for some specific data-type. E.g.,
    see ``modal_damp_vis`` at Data-set 55.
        
    **Data-set 15 (points data)**:
        
        * ``'type'``               -- *type number = 15*
        * ``'node_nums'``          -- *list of n node numbers*
        * ``'x'``                  -- *x-coordinates of the n nodes*
        * ``'y'``                  -- *y-coordinates of the n nodes*
        * ``'z'``                  -- *z-coordinates of the n nodes*
        * ``<'def_cs'>``           -- *n deformation cs numbers*
        * ``<'disp_cs'>``          -- *n displacement cs numbers*
        * ``<'color'>``            -- *n color numbers*

    **Data-set 2412 (elements data)**:
        * ``'type'``               -- *type number = 2412*
        * ``'element_nums'``       -- *list of n element numbers*
        * ``'num_nodes'``          -- *n number of nodes on element*
        * ``'nodes_nums'``         -- *n list of node numbers defining element*
        * ``'fe_descriptor'``      -- *n fe descriptor id*
        * ``'phys_table'``         -- *n physical property table number*
        * ``'mat_table'``          -- *n material property table number*
        * ``<'color'>``            -- *n color numbers*
    
    **Data-set 82 (line data)**:
        
        * ``'type'``               -- *type number = 82*
        * ``'trace_num'``          -- *number of the trace*
        * ``'lines'``              -- *list of n line numbers*
        * ``<'id'>``               -- *id string*
        * ``<'color'>``            -- *color number*
        * ``<<'n_nodes'>>``        -- *number of nodes*
        
    **Data-set 151 (header data)**:
        
        * ``'type'``               -- *type number = 151*
        * ``'model_name'``         -- *name of the model*
        * ``'description'``        -- *description of the model*
        * ``'db_app'``             -- *name of the application that
          created database*
        * ``'program'``            -- *name of the program*
        * ``'model_name'``         -- *the name of the model*
        * ``<'version_db1'>``      -- *version string 1 of the database*
        * ``<'version_db2'>``      -- *version string 2 of the database*
        * ``<'file_type'>``        -- *file type string*
        * ``<'date_db_created'>``  -- *date database was created*
        * ``<'time_db_created'>``  -- *time database was created*
        * ``<'date_db_saved'>``    -- *date database was saved*
        * ``<'time_db_saved'>``    -- *time database was saved*
        * ``<'date_file_written'>``-- *date file was written*
        * ``<'time_file_written'>``-- *time file was written*

    **Data-set 164 (units)**:
        
        * ``'type'``               -- *type number = 164*
        * ``'units_code'``         -- *units code number*
        * ``'length'``             -- *length factor*
        * ``'force'``              -- *force factor*
        * ``'temp'``               -- *temperature factor*
        * ``'temp_offset'``        -- *temperature-offset factor*
        * ``<'units_description'>``-- *units description*
        * ``<'temp_mode'>``        -- *temperature mode number*

    **Data-set 58<b> (function at nodal DOF)**:
        
        * ``'type'``               -- *type number = 58*
        * ``'func_type'``          -- *function type; only 1, 2, 3, 4
          and 6 are supported*
        * ``'rsp_node'``           -- *response node number*
        * ``'rsp_dir'``            -- *response direction number*
        * ``'ref_node'``           -- *reference node number*
        * ``'ref_dir'``            -- *reference direction number*
        * ``'data'``               -- *data array*
        * ``'x'``                  -- *abscissa array*
        * ``<'binary'>``           -- *1 for binary, 0 for ascii*
        * ``<'id1'>``              -- *id string 1*
        * ``<'id2'>``              -- *id string 2*
        * ``<'id3'>``              -- *id string 3*
        * ``<'id4'>``              -- *id string 4*
        * ``<'id5'>``              -- *id string 5*
        * ``<'load_case_id'>``     -- *id number for the load case*
        * ``<'rsp_ent_name'>``     -- *entity name for the response*
        * ``<'ref_ent_name'>``     -- *entity name for the reference*
        * ``<'abscissa_axis_units_lab'>``-- *label for the units on the abscissa*
        * ``<'abscissa_len_unit_exp'>``  -- *exp for the length unit on the abscissa*
        * ``<'abscissa_force_unit_exp'>``-- *exp for the force unit on the abscissa*
        * ``<'abscissa_temp_unit_exp'>`` -- *exp for the temperature unit on the abscissa*
        * ``<'ordinate_axis_units_lab'>``-- *label for the units on the ordinate*
        * ``<'ordinate_len_unit_exp'>``  -- *exp for the length unit on the ordinate*
        * ``<'ordinate_force_unit_exp'>``-- *exp for the force unit on the ordinate*
        * ``<'ordinate_temp_unit_exp'>`` -- *exp for the temperature unit on the ordinate*
        * ``<'orddenom_axis_units_lab'>``-- *label for the units on the ordinate denominator*
        * ``<'orddenom_len_unit_exp'>``  -- *exp for the length unit on the ordinate denominator*
        * ``<'orddenom_force_unit_exp'>``-- *exp for the force unit on the ordinate denominator*
        * ``<'orddenom_temp_unit_exp'>`` -- *exp for the temperature unit on the ordinate denominator*
        * ``<'z_axis_axis_units_lab'>``  -- *label for the units on the z axis*
        * ``<'z_axis_len_unit_exp'>``    -- *exp for the length unit on the z axis*
        * ``<'z_axis_force_unit_exp'>``  -- *exp for the force unit on the z axis*
        * ``<'z_axis_temp_unit_exp'>``   -- *exp for the temperature unit on the z axis*
        * ``<'z_axis_value'>``           -- *z axis value*
        * ``<'spec_data_type'>``         -- *specific data type*
        * ``<'abscissa_spec_data_type'>``-- *abscissa specific data type*
        * ``<'ordinate_spec_data_type'>``-- *ordinate specific data type*
        * ``<'orddenom_spec_data_type'>``-- *ordinate denominator specific data type*
        * ``<'z_axis_spec_data_type'>``  -- *z-axis specific data type*
        * ``<'ver_num'>``                -- *version number*
        * ``<<'ord_data_type'>>``        -- *ordinate data type*
        * ``<<'abscissa_min'>>``         -- *abscissa minimum*
        * ``<<'byte_ordering'>>``        -- *byte ordering*
        * ``<<'fp_format'>>``            -- *floating-point format*
        * ``<<'n_ascii_lines'>>``        -- *number of ascii lines*
        * ``<<'n_bytes'>>``              -- *number of bytes*
        * ``<<'num_pts'>>``              -- *number of data pairs for
          uneven abscissa or number of data values for even abscissa*
        * ``<<'abscissa_spacing'>>``     -- *abscissa spacing; 0=uneven,
          1=even*
        * ``<<'abscissa_inc'>>``         -- *abscissa increment; 0 if
          spacing uneven*

    **Data-set 55 (data at nodes)**:
        
        * ``'type'``               -- *type number = 55*
        * ``'analysis_type'``      -- *analysis type number; currently
          only only normal mode (2), complex eigenvalue first order
          (displacement) (3), frequency response and (5) and complex eigenvalue
          second order (velocity) (7) are supported*
        * ``'data_ch'``            -- *data-characteristic number*
        * ``'spec_data_type'``     -- *specific data type number*
        * ``'load_case'``          -- *load case number*
        * ``'mode_n'``             -- *mode number; applicable to
          analysis types 2, 3 and 7 only*
        * ``'eig'``                -- *eigen frequency (complex number);
          applicable to analysis types 3 and 7 only*
        * ``'freq'``               -- *frequency (Hz); applicable to
          analysis types 2 and 5 only*
        * ``'freq_step_n'``        -- *frequency step number; applicable
          to analysis type 5 only*
        * ``'node_nums'``          -- *node numbers*
        * ``'r1'..'r6'``           -- *response array for each DOF; when
          response is complex only r1 through r3 will be used*
        * ``<'id1'>``              -- *id1 string*
        * ``<'id2'>``              -- *id2 string*
        * ``<'id3'>``              -- *id3 string*
        * ``<'id4'>``              -- *id4 string*
        * ``<'id5'>``              -- *id5 string*
        * ``<'model_type'>``       -- *model type number*
        * ``<'modal_m'>``          -- *modal mass; applicable to
          analysis type 2 only*
        * ``<'modal_damp_vis'>``   -- *modal viscous damping ratio;
          applicable to analysis type 2 only*
        * ``<'modal_damp_his'>``   -- *modal hysteretic damping ratio;
          applicable to analysis type 2 only*
        * ``<'modal_b'>``          -- *modal-b (complex number);
          applicable to analysis types 3 and 7 only*
        * ``<'modal_a'>``          -- *modal-a (complex number);
          applicable to analysis types 3 and 7 only*
        * ``<<'n_data_per_node'>>``-- *number of data per node (DOFs)*
        * ``<<'data_type'>>``      -- *data type number; 2 = real data,
          5 = complex data*
    """

    def __init__(self, fileName):
        """
        Initializes the uff object and extract the basic info: 
        the number of sets, types of the sets and format of the sets (ascii
        or binary). To manually refresh this info, call the refresh method
        manually.
        
        Whenever some data is written to a file, a read-only flag 
        indicates that the file needs to be refreshed - before any reading,
        the file is refreshed automatically (when needed).
        """
        # Some "private" members
        self._fileName = fileName
        self._blockInd = []  # an array of block indices: start-end pairs in rows
        self._refreshed = False
        self._nSets = 0  # number of sets found in file
        self._setTypes = np.array(())  # list of set-type numbers
        self._setFormats = np.array(())  # list of set-format numbers (0=ascii,1=binary)
        # Refresh
        self.refresh()

    def get_supported_sets(self):
        """Returns a list of data-sets supported for reading and writing."""
        return _SUPPORTED_SETS

    def get_n_sets(self):
        """
        Returns the number of valid sets found in the file."""
        if not self._refreshed:
            self.refresh()
        return self._nSets

    def get_set_types(self):
        """
        Returns an array of data-set types. All valid data-sets are returned,
        even those that are not supported, i.e., whose contents will not be
        read.
        """
        if not self._refreshed:
            self.refresh()
        return self._setTypes

    def get_set_formats(self):
        """Returns an array of data-set formats: 0=ascii, 1=binary."""
        if not self._refreshed:
            self.refresh()
        return self._setFormats

    def get_file_name(self):
        """Returns the file name (as a string) associated with the uff object."""
        return self._fileName

    def file_exists(self):
        """
        Returns true if the file exists and False otherwise. If the file does
        not exist, invoking one of the read methods would raise the UFFException
        exception.
        """
        return os.path.exists(self._fileName)

    def get_status(self):
        """
        Returns the file status, i.e., True if the file is refreshed and
        False otherwise.
        """
        return self._refreshed

    def refresh(self):
        """
        Extract/refreshes the info of all the sets from UFF file (if the file
        exists). The file must exist and must be accessable otherwise, an
        error is raised. If the file cannot be refreshed, False is returned and
        True otherwise.
        """
        self._refreshed = False
        if not self.file_exists():
            return False  # cannot read the file if it does not exist
        try:
            fh = open(self._fileName, 'rb')
        #             fh = open(self._fileName, 'rt')
        except:
            raise UFFException('Cannot access the file %s' % self._fileName)
        else:
            try:
                # Parses the entire file for '    -1' tags and extracts
                # the corresponding indices
                data = fh.read()
                dataLen = len(data)
                ind = -1
                blockInd = []
                while True:
                    ind = data.find(b'    -1', ind + 1)
                    if ind == -1:
                        break
                    blockInd.append(ind)
                blockInd = np.asarray(blockInd, dtype='int64')

                # Constructs block indices of start and end values; each pair
                # points to start and end offset of the data-set (block) data,
                # but, the start '    -1' tag is included while the end one is
                # excluded.
                nBlocks = int(np.floor(len(blockInd) / 2.0))
                if nBlocks == 0:
                    # No valid blocks found but the file is still considered
                    # being refreshed
                    fh.close()
                    self._refreshed = True
                    return self._refreshed
                self._blockInd = np.zeros((nBlocks, 2), dtype='int64')
                self._blockInd[:, 0] = blockInd[:-1:2].copy()
                self._blockInd[:, 1] = blockInd[1::2].copy() - 1

                # Go through all the data-sets (blocks) and extract data-set
                # type and the property whether the data-set is in binary
                # or ascii format
                self._nSets = nBlocks
                self._setTypes = np.zeros(nBlocks)
                self._setFormats = np.zeros(nBlocks)
                for ii in range(0, self._nSets):
                    si = self._blockInd[ii, 0]
                    ei = self._blockInd[ii, 1]
                    try:
                        blockData = data[si:ei + 1].splitlines()
                        self._setTypes[ii] = int(blockData[1][0:6])
                        if blockData[1][6].lower() == 'b':
                            self._setFormats[ii] = 1
                    except:
                        # Some non-valid blocks found; ignore the exception
                        pass
                del blockInd
            except:
                fh.close()
                raise UFFException('Error refreshing UFF file: ' + self._fileName)
            else:
                self._refreshed = True
                fh.close()
                return self._refreshed

    def read_sets(self, setn=None):
        """
        Reads sets from the list or array ``setn``. If ``setn=None``, all
        sets are read (default). Sets are numbered starting at 0, ending at
        n-1. The method returns a list of dset dictionaries - as
        many dictionaries as there are sets. Unknown data-sets are returned
        empty.
        
        User must be sure that, since the last reading/writing/refreshing,
        the data has not changed by some other means than through the
        UFF object.
        """
        dset = []
        if setn is None:
            readRange = range(0, self._nSets)
        else:
            if (not type(setn).__name__ == 'list'):
                readRange = [setn]
            else:
                readRange = setn
        if not self.file_exists():
            raise UFFException('Cannot read from a non-existing file: ' + self._fileName)
        if not self._refreshed:
            if not self._refresh():
                raise UFFException('Cannot read from the file: ' + self._fileName)
        try:
            for ii in readRange:
                dset.append(self._read_set(ii))
        except UFFException as msg:
            raise UFFException('Error when reading ' + str(ii) + '-th data-set: ' + msg.value)
        except:
            raise UFFException('Error when reading data-set(s)')
        if len(dset) == 1:
            dset = dset[0]
        return dset

    ##    def read_all_sets(self):
    ##        """Reads all the sets from UFF file.
    ##        The method returns a list of dsets each containing as many
    ##        dictionaries as there are valid sets found in the file.
    ##        """
    ##        dset = []
    ##        if not self.file_exists():
    ##            raise UFFException('Cannot read from a non-existing file: '+self._fileName)
    ##        if not self._refreshed:
    ##            if not self._refresh():
    ##                raise UFFException('Cannot read from the file: '+self._fileName)
    ##        try:
    ##            for ii in range(0,self._nSets):
    ##                dset.append(self._read_set(ii))
    ##        except UFFException,msg:
    ##            raise UFFException('Error when reading '+str(ii)+'-th data-set: '+msg.value)
    ##        except:
    ##            raise UFFException('Error when reading '+str(ii)+'-th data-set')
    ##        return dset

    def write_sets(self, dsets, mode='add'):
        """
        Writes several UFF data-sets to the file.  The mode can be
        either 'add' (default) or 'overwrite'. The dsets is a
        list of dictionaries, each representing one data-set. Unsupported
        data-sets will be ignored. When only 1 data-set is to be written, no
        lists are necessary, i.e., only one dictionary is required.

        For each data-set, there are some optional and some required fields at
        dset dictionary. Also, in general, the sum of the required
        and the optional fields together can be less then the number of fields
        read from the same type of data-set; the reason is that for some
        data-sets some fields are set automatically. Optional fields are
        calculated automatically and the dset is updated - as dset is actually
        an alias (aka pointer), this is reflected at the caller too.
        """
        if (not type(dsets).__name__ == 'list'):
            dsets = [dsets]
        nSets = len(dsets)
        if nSets < 1:
            raise UFFException('Nothing to write')
        if mode.lower() == 'overwrite':
            # overwrite mode; first set is written in the overwrite mode, others
            # in add mode
            self._write_set(dsets[0], 'overwrite')
            for ii in range(1, nSets):
                self._write_set(dsets[ii], 'add')
        elif mode.lower() == 'add':
            # add mode; all the sets are written in the add mode
            for ii in range(0, nSets):
                self._write_set(dsets[ii], 'add')
        else:
            raise UFFException('Unknown mode: ' + mode)

    def _read_set(self, n):
        # Reads n-th set from UFF file. n can be an integer between 0 and nSets-1.
        # User must be sure that, since the last reading/writing/refreshing,
        # the data has not changed by some other means than through the
        # UFF object. The method returns dset dictionary.
        dset = {}
        if not self.file_exists():
            raise UFFException('Cannot read from a non-existing file: ' + self._fileName)
        if not self._refreshed:
            if not self.refresh():
                raise UFFException('Cannot read from the file: ' + self._fileName + '. The file cannot be refreshed.')
        if (n > self._nSets - 1) or (n < 0):
            raise UFFException('Cannot read data-set: ' + str(int(n)) + \
                               '. Data-set number to high or to low.')
        # Read n-th data-set data (one block)
        try:
            fh = open(self._fileName, 'rb')
        except:
            raise UFFException('Cannot access the file: ' + self._fileName + ' to read from.')
        else:
            try:
                si = self._blockInd[n][0]  # start offset
                ei = self._blockInd[n][1]  # end offset
                fh.seek(si)
                if self._setTypes[int(n)] == 58:
                    blockData = fh.read(ei - si + 1)  # decoding is handled later in _extract58
                else:
                    blockData = fh.read(ei - si + 1).decode('utf-8', errors='replace')
            except:
                fh.close()
                raise UFFException('Error reading data-set #: ' + int(n))
            else:
                fh.close()
        # Extracts the dset
        if self._setTypes[int(n)] == 15:
            dset = self._extract15(blockData)
        elif self._setTypes[int(n)] == 18:
            dset = self._extract18(blockData)  # TEMP ADD
        elif self._setTypes[int(n)] == 55:
            dset = self._extract55(blockData)
        elif self._setTypes[int(n)] == 58:
            dset = self._extract58(blockData)
        elif self._setTypes[int(n)] == 82:
            dset = self._extract82(blockData)
        elif self._setTypes[int(n)] == 151:
            dset = self._extract151(blockData)
        elif self._setTypes[int(n)] == 164:
            dset = self._extract164(blockData)
        elif self._setTypes[int(n)] == 2411:
            dset = self._extract2411(blockData)  # TEMP ADD
        elif self._setTypes[int(n)] == 2412:
            dset = self._extract2412(blockData)
        elif self._setTypes[int(n)] == 2414:
            dset = self._extract2414(blockData) 
        elif self._setTypes[int(n)] == 2420:
            dset = self._extract2420(blockData)
        else:
            dset['type'] = self._setTypes[int(n)]
            # Unsupported data-set - do nothing
            pass
        return dset

    def _write_set(self, dset, mode='add'):
        # Writes UFF data (UFF data-sets) to the file.  The mode can be
        # either 'add' (default) or 'overwrite'. The dset is a
        # dictionary of keys and corresponding values. Unsupported
        # data-set will be ignored.
        # 
        # For each data-set, there are some optional and some required fields at
        # dset dictionary. Also, in general, the sum of the required
        # and the optional fields together can be less then the number of fields
        # read from the same type of data-set; the reason is that for some
        # data-sets some fields are set automatically. Optional fields are
        # calculated automatically and the dset is updated - as dset is actually
        # an alias (aka pointer), this is reflected at the caller too.
        if mode.lower() == 'overwrite':
            # overwrite mode
            try:
                fh = open(self._fileName, 'wt')
            except:
                raise UFFException('Cannot access the file: ' + self._fileName + ' to write to.')
        elif mode.lower() == 'add':
            # add (append) mode
            try:
                fh = open(self._fileName, 'at')
            except:
                raise UFFException('Cannot access the file: ' + self._fileName + ' to write to.')
        else:
            raise UFFException('Unknown mode: ' + mode)
        try:
            # Actual writing
            try:
                setType = dset['type']
            except:
                fh.close()
                raise UFFException('Data-set\'s dictionary is missing the required \'type\' key')
            # handle nan or inf
            if 'data' in dset.keys():
                dset['data'] = np.nan_to_num(dset['data'])

            if setType == 15:
                self._write15(fh, dset)
            elif setType == 55:
                self._write55(fh, dset)
            elif setType == 58:
                self._write58(fh, dset, mode)
            elif setType == 82:
                self._write82(fh, dset)
            elif setType == 151:
                self._write151(fh, dset)
            elif setType == 164:
                self._write164(fh, dset)
            elif setType == 2411:
                self._write2411(fh, dset)
            elif setType == 2412:
                self._write2412(fh, dset)
            elif setType == 2414:
                self._write2414(fh, dset)
            elif setType == 2420:
                self._write2420(fh, dset)
            else:
                # Unsupported data-set - do nothing
                pass
        except:
            fh.close()
            raise  # re-raise the last exception
        else:
            fh.close()
        self.refresh()

    def _write15(self, fh, dset):
        # Writes coordinate data - data-set 15 - to an open file fh
        try:
            n = len(dset['node_nums'])
            # handle optional fields
            dset = self._opt_fields(dset, {'def_cs': np.asarray([0 for ii in range(0, n)], 'i'),
                                           'disp_cs': np.asarray([0 for ii in range(0, n)], 'i'),
                                           'color': np.asarray([0 for ii in range(0, n)], 'i')})
            # write strings to the file
            fh.write('%6i\n%6i%74s\n' % (-1, 15, ' '))
            for ii in range(0, n):
                fh.write('%10i%10i%10i%10i%13.5e%13.5e%13.5e\n' % (
                    dset['node_nums'][ii], dset['def_cs'][ii], dset['disp_cs'][ii], dset['color'][ii],
                    dset['x'][ii], dset['y'][ii], dset['z'][ii]))
            fh.write('%6i\n' % -1)
        except KeyError as msg:
            raise UFFException('The required key \'' + msg.args[0] + '\' not present when writing data-set #15')
        except:
            raise UFFException('Error writing data-set #15')

    def _write82(self, fh, dset):
        # Writes line data - data-set 82 - to an open file fh
        try:
            # handle optional fields
            dset = self._opt_fields(dset, {'id': 'NONE',
                                           'color': 0})
            # write strings to the file
            # removed jul 2017: unique_nodes = set(dset['nodes'])
            # removed jul 2017:if 0 in unique_nodes: unique_nodes.remove(0)
            # number of changes of node need to
            # nNodes = len(dset['nodes'])
            nNodes = np.sum((dset['nodes'][1:] - dset['nodes'][:-1]) != 0) + 1
            fh.write('%6i\n%6i%74s\n' % (-1, 82, ' '))
            fh.write('%10i%10i%10i\n' % (dset['trace_num'], nNodes, dset['color']))
            fh.write('%-80s\n' % dset['id'])
            sl = 0
            n8Blocks = nNodes // 8
            remLines = nNodes % 8
            if n8Blocks:
                for ii in range(0, n8Blocks):
                    #                 fh.write( string.join(['%10i'%lineN for lineN in dset['lines'][sl:sl+8]],'')+'\n' )
                    fh.write(''.join(['%10i' % lineN for lineN in dset['nodes'][sl:sl + 8]]) + '\n')
                    sl += 8
            if remLines > 0:
                fh.write(''.join(['%10i' % lineN for lineN in dset['nodes'][sl:]]) + '\n')
            #                 fh.write( string.join(['%10i'%lineN for lineN in dset['lines'][sl:]],'')+'\n' )
            fh.write('%6i\n' % -1)
        except KeyError as msg:
            raise UFFException('The required key \'' + msg.args[0] + '\' not present when writing data-set #82')
        except:
            raise UFFException('Error writing data-set #82')

    def _write151(self, fh, dset):
        # Writes dset data - data-set 151 - to an open file fh
        try:
            ds = time.strftime('%d-%b-%y', time.localtime())
            ts = time.strftime('%H:%M:%S', time.localtime())
            # handle optional fields
            dset = self._opt_fields(dset, {'version_db1': '0',
                                           'version_db2': '0',
                                           'file_type': '0',
                                           'date_db_created': ds,
                                           'time_db_created': ts,
                                           'date_db_saved': ds,
                                           'time_db_saved': ts,
                                           'date_file_written': ds,
                                           'time_file_written': ts})
            # write strings to the file
            fh.write('%6i\n%6i%74s\n' % (-1, 151, ' '))
            fh.write('%-80s\n' % dset['model_name'])
            fh.write('%-80s\n' % dset['description'])
            fh.write('%-80s\n' % dset['db_app'])
            fh.write('%-10s%-10s%10s%10s%10s\n' % (dset['date_db_created'],
                                                   dset['time_db_created'], dset['version_db1'], dset['version_db2'],
                                                   dset['file_type']))
            fh.write('%-10s%-10s\n' % (dset['date_db_saved'], dset['time_db_saved']))
            fh.write('%-80s\n' % dset['program'])
            fh.write('%-10s%-10s\n' % (dset['date_file_written'], dset['time_file_written']))
            fh.write('%6i\n' % -1)
        except KeyError as msg:
            raise UFFException('The required key \'' + msg.args[0] + '\' not present when writing data-set #151')
        except:
            raise UFFException('Error writing data-set #151')

    def _write164(self, fh, dset):
        # Writes units data - data-set 164 - to an open file fh
        try:
            # handle optional fields
            dset = self._opt_fields(dset, {'units_description': 'User unit system',
                                           'temp_mode': 1})
            # write strings to the file
            fh.write('%6i\n%6i%74s\n' % (-1, 164, ' '))
            fh.write('%10i%20s%10i\n' % (dset['units_code'], dset['units_description'], dset['temp_mode']))
            str = '%25.16e%25.16e%25.16e\n%25.16e\n' % (
                dset['length'], dset['force'], dset['temp'], dset['temp_offset'])
            str = str.replace('e+', 'D+')
            str = str.replace('e-', 'D-')
            fh.write(str)
            fh.write('%6i\n' % -1)
        except KeyError as msg:
            raise UFFException('The required key \'' + msg.args[0] + '\' not present when writing data-set #164')
        except:
            raise UFFException('Error writing data-set #164')

    def _write55(self, fh, dset):
        # Writes data at nodes - data-set 55 - to an open file fh. Currently:
        #   - only normal mode (2)
        #   - complex eigenvalue first order (displacement) (3)
        #   - frequency response and (5)
        #   - complex eigenvalue second order (velocity) (7)
        # analyses are supported.
        try:
            # Handle general optional fields
            dset = self._opt_fields(dset,
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
                dset = self._opt_fields(dset,
                                        {'modal_m': 0,
                                         'modal_damp_vis': 0,
                                         'modal_damp_his': 0})
            elif dset['analysis_type'] in (3, 7):
                # complex modes
                dset = self._opt_fields(dset,
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
                raise UFFException('Error writing data-set #55: unsupported analysis type')
            # Some additional checking
            dataType = 2
            #             if dset.has_key('r4') and dset.has_key('r5') and dset.has_key('r6'):
            if ('r4' in dset) and ('r5' in dset) and ('r6' in dset):
                nDataPerNode = 6
            else:
                nDataPerNode = 3
            if np.iscomplexobj(dset['r1']):
                dataType = 5
            else:
                dataType = 2
            # Write strings to the file
            fh.write('%6i\n%6i%74s\n' % (-1, 55, ' '))
            fh.write('%-80s\n' % dset['id1'])
            fh.write('%-80s\n' % dset['id2'])
            fh.write('%-80s\n' % dset['id3'])
            fh.write('%-80s\n' % dset['id4'])
            fh.write('%-80s\n' % dset['id5'])
            fh.write('%10i%10i%10i%10i%10i%10i\n' %
                     (dset['model_type'], dset['analysis_type'], dset['data_ch'],
                      dset['spec_data_type'], dataType, nDataPerNode))
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
                raise UFFException('Unsupported analysis type')
            n = len(dset['node_nums'])
            if dataType == 2:
                # Real data
                if nDataPerNode == 3:
                    for k in range(0, n):
                        fh.write('%10i\n' % dset['node_nums'][k])
                        fh.write('%13.5e%13.5e%13.5e\n' % (dset['r1'][k], dset['r2'][k], dset['r3'][k]))
                else:
                    for k in range(0, n):
                        fh.write('%10i\n' % dset['node_nums'][k])
                        fh.write('%13.5e%13.5e%13.5e%13.5e%13.5e%13.5e\n' %
                                 (dset['r1'][k], dset['r2'][k], dset['r3'][k], dset['r4'][k], dset['r5'][k],
                                  dset['r6'][k]))
            elif dataType == 5:
                # Complex data; n_data_per_node is assumed being 3
                for k in range(0, n):
                    fh.write('%10i\n' % dset['node_nums'][k])
                    fh.write('%13.5e%13.5e%13.5e%13.5e%13.5e%13.5e\n' %
                             (dset['r1'][k].real, dset['r1'][k].imag, dset['r2'][k].real, dset['r2'][k].imag,
                              dset['r3'][k].real, dset['r3'][k].imag))
            else:
                raise UFFException('Unsupported data type')
            fh.write('%6i\n' % -1)
        except KeyError as msg:
            raise UFFException('The required key \'' + msg.args[0] + '\' not present when writing data-set #55')
        except:
            raise UFFException('Error writing data-set #55')

    def _write58(self, fh, dset, mode='add'):
        # Writes function at nodal DOF - data-set 58 - to an open file fh.
        try:
            if not (dset['func_type'] in [1, 2, 3, 4, 6]):
                raise UFFException('Unsupported function type')
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
            dset = self._opt_fields(dset, dict)
            # Write strings to the file - always in double precision => ord_data_type = 2
            # for real data and 6 for complex data
            numPts = len(dset['data'])
            isR = not np.iscomplexobj(dset['data'])
            if isR:
                # real data
                dset['ord_data_type'] = 4
                nBytes = numPts * 8
                if 'n_bytes' in dset.keys():
                    dset['n_bytes'] = nBytes
                ordDataType = dset['ord_data_type']
            else:
                # complex data
                dset['ord_data_type'] = 6
                nBytes = numPts * 8
                ordDataType = 6

            isEven = bool(dset['abscissa_spacing'])  # handling even/uneven abscissa spacing manually

            # handling abscissa spacing automatically
            # isEven = len( set( [ dset['x'][ii]-dset['x'][ii-1] for ii in range(1,len(dset['x'])) ] ) ) == 1
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
                fh.write('b%6i%6i%12i%12i%6i%6i%12i%12i\n' % (bo, 2, 11, nBytes, 0, 0, 0, 0))
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
            fh.write('%10i%10i%10i%13.5e%13.5e%13.5e\n' % (ordDataType, numPts, isEven,
                                                           isEven * dset['abscissa_min'], isEven * dx,
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
            if isR:
                if isEven:
                    data = dset['data'].copy()
                else:
                    data = np.zeros(2 * numPts, 'd')
                    data[0:-1:2] = dset['x']
                    data[1::2] = dset['data']
            else:
                if isEven:
                    data = np.zeros(2 * numPts, 'd')
                    data[0:-1:2] = dset['data'].real
                    data[1::2] = dset['data'].imag
                else:
                    data = np.zeros(3 * numPts, 'd')
                    data[0:-2:3] = dset['x']
                    data[1:-1:3] = dset['data'].real
                    data[2::3] = dset['data'].imag
            # always write data in double precision
            if dset['binary']:
                fh.close()
                if mode.lower() == 'overwrite':
                    fh = open(self._fileName, 'wb')
                elif mode.lower() == 'add':
                    fh = open(self._fileName, 'ab')
                # write data
                if bo == 1:
                    [fh.write(struct.pack('<d', datai)) for datai in data]
                else:
                    [fh.write(struct.pack('>d', datai)) for datai in data]
                fh.close()
                if mode.lower() == 'overwrite':
                    fh = open(self._fileName, 'wt')
                elif mode.lower() == 'add':
                    fh = open(self._fileName, 'at')
            else:
                n4Blocks = len(data) // 4
                remVals = len(data) % 4
                if isR:
                    if isEven:
                        fh.write(n4Blocks * '%20.11e%20.11e%20.11e%20.11e\n' % tuple(data[:4 * n4Blocks]))
                        if remVals > 0:
                            fh.write((remVals * '%20.11e' + '\n') % tuple(data[4 * n4Blocks:]))
                    else:
                        fh.write(n4Blocks * '%13.5e%20.11e%13.5e%20.11e\n' % tuple(data[:4 * n4Blocks]))
                        if remVals > 0:
                            fmt = ['%13.5e', '%20.11e', '%13.5e', '%20.11e']
                            fh.write((''.join(fmt[remVals]) + '\n') % tuple(data[4 * n4Blocks:]))
                else:
                    if isEven:
                        fh.write(n4Blocks * '%20.11e%20.11e%20.11e%20.11e\n' % tuple(data[:4 * n4Blocks]))
                        if remVals > 0:
                            fh.write((remVals * '%20.11e' + '\n') % tuple(data[4 * n4Blocks:]))
                    else:
                        n3Blocks = len(data) / 3
                        remVals = len(data) % 3
                        # TODO: It breaks here for long measurements. Implement exceptions.
                        # n3Blocks seems to be a natural number but of the wrong type. Convert for now,
                        # but make assertion to prevent werid things from happening.
                        if float(n3Blocks - int(n3Blocks)) != 0.0:
                            print('Warning: Something went wrong when savning the uff file.')
                        n3Blocks = int(n3Blocks)
                        fh.write(n3Blocks * '%13.5e%20.11e%20.11e\n' % tuple(data[:3 * n3Blocks]))
                        if remVals > 0:
                            fmt = ['%13.5e', '%20.11e', '%20.11e']
                            fh.write((''.join(fmt[remVals]) + '\n') % tuple(data[3 * n3Blocks:]))
            fh.write('%6i\n' % -1)
            del data
        except KeyError as msg:
            raise UFFException('The required key \'' + msg.args[0] + '\' not present when writing data-set #58')
        except:
            raise UFFException('Error writing data-set #58')

    def _write2411(self, fh, dset):
        try:
            dict = {'export_cs_number': 0,
                    'cs_color': 8}

            dset = self._opt_fields(dset, dict)
            fh.write('%6i\n%6i%74s\n' % (-1, 2411, ' '))

            for node in range(dset['grid_global'].shape[0]):
                fh.write('%10i%10i%10i%10i\n' % (dset['grid_global'][node, 0], dset['export_cs_number'],
                                                 dset['grid_global'][node, 0], dset['cs_color']))

                fh.write('%25.16e%25.16e%25.16e\n' % tuple(dset['grid_global'][node, 1:]))

            fh.write('%6i\n' % -1)

        except:
            raise UFFException('Error writing data-set #2411')

    def _write2412(self, fh, dset):
        try:
            elt_type_dict = {'triangle': 3, 'quad': 4}
            fh.write('%6i\n%6i%74s\n' % (-1, 2412, ' '))
            for elt_type in dset:
                if elt_type == "type":
                    pass
                else:
                    for i in range(len(dset[elt_type]['element_nums'])):
                        fh.write('%10i%10i%10i%10i%10i%10i\n' % (
                            dset[elt_type]['element_nums'][i],
                            dset[elt_type]['fe_descriptor'][i],
                            dset[elt_type]['phys_table'][i],
                            dset[elt_type]['mat_table'][i],
                            dset[elt_type]['color'][i],
                            elt_type_dict[elt_type],
                        ))
                        for ii in range(elt_type_dict[elt_type]):
                            fh.write('%10i' % dset[elt_type]['nodes_nums'][i][ii])
                        fh.write('\n')
            fh.write('%6i\n' % -1)

        except:
            raise UFFException('Error writing data-set #2412')

    def _write2414(self, fh, dset):
        #DS2414_num is iterative number for each DS2414
        #Nthfreq is th frequency
        #Writes data at nodes - data-set 2414 - to an open file fh. Currently:
        #   - frequency response (5)
        # analyses are supported.
        try:
            # Handle general optional fields
            
            if dset['analysis_type']==5:
                fh.write('%6i\n%6i\n' % (-1, 2414))
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
                for node in range(dset['node_nums'].shape[0]):
                    fh.write('%10i\n' % (int(dset['node_nums'][node])))
                    fh.write('%13.5e%13.5e%13.5e%13.5e%13.5e%13.5e\n' % (
                                                np.real(dset['x'][node]),
                                                np.imag(dset['x'][node]),
                                                np.real(dset['y'][node]),
                                                np.imag(dset['y'][node]),
                                                np.real(dset['z'][node]),
                                                np.imag(dset['z'][node])))
                fh.write('%6i\n' % (-1))    
        except:
            raise UFFException('Error writing data-set #2414')

    # TODO: Big deal - the output dictionary when reading this set
    #    is different than the dictionary that is expected (keys) when
    #    writing this same set. This is not OK!
    def _write2420(self, fh, dset):
        try:
            dict = {'part_UID': 1,
                    'part_name': 'None',
                    'cs_type': 0,
                    'cs_color': 8}
            dset = self._opt_fields(dset, dict)

            fh.write('%6i\n%6i%74s\n' % (-1, 2420, ' '))
            fh.write('%10i\n' % (dset['part_UID']))
            fh.write('%-80s\n' % (dset['part_name']))

            for node in range(len(dset['nodes'])):
                fh.write('%10i%10i%10i\n' % (dset['nodes'][node], dset['cs_type'], dset['cs_color']))
                fh.write('CS%i\n' % dset['nodes'][node])
                fh.write('%25.16e%25.16e%25.16e\n' % tuple(dset['local_cs'][node * 4, :]))
                fh.write('%25.16e%25.16e%25.16e\n' % tuple(dset['local_cs'][node * 4 + 1, :]))
                fh.write('%25.16e%25.16e%25.16e\n' % tuple(dset['local_cs'][node * 4 + 2, :]))
                fh.write('%25.16e%25.16e%25.16e\n' % tuple(dset['local_cs'][node * 4 + 3, :]))
            fh.write('%6i\n' % -1)
        except:
            raise UFFException('Error writing data-set #2420')

    def _extract15(self, blockData):
        # Extract coordinate data - data-set 15.
        dset = {'type': 15}
        try:
            # Body
            split_data = blockData.splitlines()
            split_data = ''.join(split_data[2:]).split()
            split_data = [float(_) for _ in split_data]

            dset['node_nums'] = split_data[::7]
            dset['def_cs'] = split_data[1::7]
            dset['disp_cs'] = split_data[2::7]
            dset['color'] = split_data[3::7]
            dset['x'] = split_data[4::7]
            dset['y'] = split_data[5::7]
            dset['z'] = split_data[6::7]
        except:
            raise UFFException('Error reading data-set #15')
        return dset

    def _extract2411(self, blockData):
        # Extract coordinate data - data-set 15.
        dset = {'type': 15}
        try:
            # Body
            splitData = blockData.splitlines(True)  # Keep the line breaks!
            splitData = ''.join(splitData[2:])  # ..as they are again needed
            splitData = splitData.split()
            values = np.asarray([float(str) for str in splitData], 'd')
            dset['node_nums'] = values[::7].copy()
            dset['def_cs'] = values[1::7].copy()
            dset['disp_cs'] = values[2::7].copy()
            dset['color'] = values[3::7].copy()
            dset['x'] = values[4::7].copy()
            dset['y'] = values[5::7].copy()
            dset['z'] = values[6::7].copy()
        except:
            raise UFFException('Error reading data-set #15')
        return dset

    def _extract2412(self,blockData):
        # Extract element data - data-set 2412.
        dset = {'type': 2412}
        # Define dictionary of possible elements types
        # Only 2D non-quadratic elements are supported
        elt_type_dict = {'3': 'triangle', '4': 'quad'}
        # Read data
        try:
            splitData = blockData.splitlines()
            splitData = [a.split() for a in splitData][2:]
            # Extract Record 1
            rec1 = np.array(splitData[::2], dtype=int) 
            # Extract Record 2
            rec2 = splitData[1::2] 
            # Look for the different types of elements stored in the dataset
            elts_types = list(set(rec1[:,5]))
            for elt_type in elts_types:
                ind = np.where(np.array(rec1[:,5]) == elt_type)[0]
                dict_tmp = dict()
                dict_tmp['element_nums'] = rec1[ind,0].copy()
                dict_tmp['fe_descriptor'] = rec1[ind,1].copy()
                dict_tmp['phys_table'] = rec1[ind,2].copy()
                dict_tmp['mat_table'] = rec1[ind,3].copy()
                dict_tmp['color'] = rec1[ind,4].copy()
                dict_tmp['nodes_nums'] =  np.array([rec2[i] for i in ind], dtype=int).copy().reshape((-1,elt_type))
                dset[elt_type_dict[str(elt_type)]] = dict_tmp
        except:
            raise UFFException('Error reading data-set #2412')
        return dset
    
    def _extract2414(self,blockData):
        # Extract analysis data - data-set 2414.
        dset = {'type': 2414}
        # Read data
        try:
            binary = False
            split_header = blockData.splitlines(True)[:15]  # Keep the line breaks!
            dset.update(self._parse_header_line(split_header[2], 1, [80], [2], ['analysis_dataset_label'])) #Loadcase number
            dset.update(self._parse_header_line(split_header[3], 1, [80], [1], ['analysis_dataset_name'])) # usually with the frequency
            dset.update(self._parse_header_line(split_header[4], 1, [80], [2], ['dataset_location']))
            dset.update(self._parse_header_line(split_header[5], 1, [80], [1], ['id1']))
            dset.update(self._parse_header_line(split_header[6], 1, [80], [1], ['id2']))
            dset.update(self._parse_header_line(split_header[7], 1, [80], [1], ['id3']))  # usually with the frequency
            dset.update(self._parse_header_line(split_header[8], 1, [80], [1], ['id4']))  # usually with the loadcase
            dset.update(self._parse_header_line(split_header[9], 1, [80], [1], ['id5']))
           
            dset.update(self._parse_header_line(split_header[10], 6, [10, 10, 10, 10, 10, 10], [2, 2, 2, 2, 2, 2],
                                                ['model_type', 'analysis_type', 'data_characteristic', 'result_type',
                                                 'data_type', 'number_of_data_values_for_the_data_component']))     
            
            dset.update(self._parse_header_line(split_header[11], 8, [10, 10, 10, 10, 10, 10, 10, 10], [2, 2, 2, 2, 2, 2, 2, 2],
                                                ['design_set_id', 'iteration_number', 'solution_set_id', 'boundary_condition', 
                                                 'load_set', 'mode_number', 'time_step_number', 'frequency_number']))
            dset.update(self._parse_header_line(split_header[12], 2, [10, 10], [2, 2],
                                                ['creation_option', 'number_retained']))

            dset.update(self._parse_header_line(split_header[13], 6, [13, 13, 13, 13, 13, 13], [0.5,0.5, 0.5, 0.5, 0.5, 0.5],
                                            ['time', 'frequency', 'eigenvalue', 'modal_mass', 'viscous_damping', 'hysteretic_damping']))
            dset.update(self._parse_header_line(split_header[14], 6, [13, 13, 13, 13, 13, 13], [0.5,0.5, 0.5, 0.5, 0.5, 0.5],
                                            ['real_part_eigenvalue', 'imaginary_part_eigenvalue', 
                                                'real_part_of_modal_A_or_modal_mass', 'imaginary_part_of_modal_A_or_modal_mass', 
                                                'real_part_of_modal_B_or_modal_mass', 'imaginary_part_of_modal_B_or_modal_mass']))
            if dset['analysis_type'] == 5:
                # frequency response 
                splitData = ''.join(blockData.splitlines(True)[15:])
                splitData = splitData.split()
                if dset['data_type'] == 5 and dset['number_of_data_values_for_the_data_component'] == 3:
                    values = np.asarray([float(str) for str in splitData], 'd')
                    dset['node_nums'] = np.array(values[::7].copy(), dtype=int)
                    dset['x'] = values[1::7].copy()+values[2::7].copy()*1j
                    dset['y'] = values[3::7].copy()+values[4::7].copy()*1j
                    dset['z'] = values[5::7].copy()+values[6::7].copy()*1j   

            pass  
        except:
            raise UFFException('Error reading data-set #2412')
        return dset

    def _extract18(self, blockData):
        '''Extract local CS definitions -- data-set 18.'''
        dset = {'type': 18}
        try:
            splitData = blockData.splitlines()

            # -- Get Record 1
            rec_1 = np.array(list(map(float, ''.join(splitData[2::4]).split())))

            dset['cs_num'] = rec_1[::5]
            # removed - clutter
            # dset['cs_type'] = rec_1[1::5]
            dset['ref_cs_num'] = rec_1[2::5]
            # left out here are the parameters color and definition type

            # -- Get Record 2
            # removed because clutter
            # dset['cs_name'] = splitData[3::4]

            # -- Get Record 31 and 32
            # ... these are the origins of cs defined in ref
            #             rec_31 = np.array(list(map(float, ''.join(splitData[4::4]).split())))
            lineData = ''.join(splitData[4::4])
            rec_31 = [float(lineData[i * 13:(i + 1) * 13]) for i in range(int(len(lineData) / 13))]
            dset['ref_o'] = np.vstack((np.array(rec_31[::6]),
                                       np.array(rec_31[1::6]),
                                       np.array(rec_31[2::6]))).transpose()

            # ... these are points on the x axis of cs defined in ref
            dset['x_point'] = np.vstack((np.array(rec_31[3::6]),
                                         np.array(rec_31[4::6]),
                                         np.array(rec_31[5::6]))).transpose()

            # ... these are the points on the xz plane
            lineData = ''.join(splitData[5::4])
            rec_32 = [float(lineData[i * 13:(i + 1) * 13]) for i in range(int(len(lineData) / 13))]
            #             rec_32 = np.array(list(map(float, ''.join(splitData[5::4]).split())))
            dset['xz_point'] = np.vstack((np.array(rec_32[::3]),
                                          np.array(rec_32[1::3]),
                                          np.array(rec_32[2::3]))).transpose()
        except:
            raise UFFException('Error reading data-set #18')
        return dset

    def _extract2420(self, blockData):
        '''Extract local CS/transforms -- data-set 2420.'''
        dset = {'type': 2420}
        #        try:
        splitData = blockData.splitlines(True)

        # -- Get Record 1
        dset['Part_UID'] = float(splitData[2])

        # -- Get Record 2
        dset['Part_Name'] = splitData[3].rstrip()

        # -- Get Record 3
        rec_3 = list(map(int, ''.join(splitData[4::6]).split()))
        dset['CS_sys_labels'] = rec_3[::3]
        dset['CS_types'] = rec_3[1::3]
        dset['CS_colors'] = rec_3[2::3]

        # -- Get Record 4
        dset['CS_names'] = list(map(str.rstrip, splitData[5::6]))

        # !! The following part should be made smoother
        # -- Get Record 5
        row1 = list(map(float, ''.join(splitData[6::6]).split()))
        row2 = list(map(float, ''.join(splitData[7::6]).split()))
        row3 = list(map(float, ''.join(splitData[8::6]).split()))
        # !! Row 4 left out for now - usually zeros ...
        #            row4 = map(float, splitData[7::6].split())
        dset['CS_matrices'] = [np.vstack((row1[i:(i + 3)], row2[i:(i + 3)], row3[i:(i + 3)])) \
                               for i in np.arange(0, len(row1), 3)]
        #        except:
        #            raise UFFException('Error reading data-set #2420')
        return dset

    def _extract82(self, blockData):
        # Extract line data - data-set 82.
        dset = {'type': 82}
        try:
            splitData = blockData.splitlines(True)
            dset.update(
                self._parse_header_line(splitData[2], 3, [10, 10, 10], [2, 2, 2], ['trace_num', 'n_nodes', 'color']))
            dset.update(self._parse_header_line(splitData[3], 1, [80], [1], ['id']))
            splitData = ''.join(splitData[4:])
            splitData = splitData.split()
            dset['nodes'] = np.asarray([float(str) for str in splitData])
        except:
            raise UFFException('Error reading data-set #82')
        return dset

    def _extract151(self, blockData):
        # Extract dset data - data-set 151.
        dset = {'type': 151}
        try:
            splitData = blockData.splitlines(True)
            dset.update(self._parse_header_line(splitData[2], 1, [80], [1], ['model_name']))
            dset.update(self._parse_header_line(splitData[3], 1, [80], [1], ['description']))
            dset.update(self._parse_header_line(splitData[4], 1, [80], [1], ['db_app']))
            dset.update(self._parse_header_line(splitData[5], 2, [10, 10, 10, 10, 10], [1, 1, 2, 2, 2],
                                                ['date_db_created', 'time_db_created', 'version_db1', 'version_db2',
                                                 'file_type']))
            dset.update(self._parse_header_line(splitData[6], 1, [10, 10], [1, 1], ['date_db_saved', 'time_db_saved']))
            dset.update(self._parse_header_line(splitData[7], 1, [80], [1], ['program']))
            dset.update(
                self._parse_header_line(splitData[8], 1, [10, 10], [1, 1], ['date_file_written', 'time_file_written']))
        except:
            raise UFFException('Error reading data-set #151')
        return dset

    def _extract164(self, blockData):
        # Extract units data - data-set 164.
        dset = {'type': 164}
        try:
            splitData = blockData.splitlines(True)
            dset.update(self._parse_header_line(splitData[2], 1, [10, 20, 10], [2, 1, 2],
                                                ['units_code', 'units_description', 'temp_mode']))
            splitData = ''.join(splitData[3:])
            splitData = splitData.split()
            dset['length'] = float(splitData[0].lower().replace('d', 'e'))
            dset['force'] = float(splitData[1].lower().replace('d', 'e'))
            dset['temp'] = float(splitData[2].lower().replace('d', 'e'))
            dset['temp_offset'] = float(splitData[3].lower().replace('d', 'e'))
        except:
            raise UFFException('Error reading data-set #164')
        return dset

    def _extract55(self, blockData):
        # Extract data at nodes - data-set 55. Currently:
        #   - only normal mode (2)
        #   - complex eigenvalue first order (displacement) (3)
        #   - frequency response and (5)
        #   - complex eigenvalue second order (velocity) (7)
        # analyses are supported.
        dset = {'type': 55}
        try:
            splitData = blockData.splitlines(True)
            dset.update(self._parse_header_line(splitData[2], 1, [80], [1], ['id1']))
            dset.update(self._parse_header_line(splitData[3], 1, [80], [1], ['id2']))
            dset.update(self._parse_header_line(splitData[4], 1, [80], [1], ['id3']))
            dset.update(self._parse_header_line(splitData[5], 1, [80], [1], ['id4']))
            dset.update(self._parse_header_line(splitData[6], 1, [80], [1], ['id5']))
            dset.update(self._parse_header_line(splitData[7], 6, [10, 10, 10, 10, 10, 10], [2, 2, 2, 2, 2, 2],
                                                ['model_type', 'analysis_type', 'data_ch', 'spec_data_type',
                                                 'data_type', 'n_data_per_node']))
            if dset['analysis_type'] == 2:
                # normal mode
                dset.update(self._parse_header_line(splitData[8], 4, [10, 10, 10, 10, 10, 10, 10, 10],
                                                    [-1, -1, 2, 2, -1, -1, -1, -1],
                                                    ['', '', 'load_case', 'mode_n', '', '', '', '']))
                dset.update(self._parse_header_line(splitData[9], 4, [13, 13, 13, 13, 13, 13], [3, 3, 3, 3, -1, -1],
                                                    ['freq', 'modal_m', 'modal_damp_vis', 'modal_damp_his', '', '']))
            elif (dset['analysis_type'] == 3) or (dset['analysis_type'] == 7):
                # complex eigenvalue
                dset.update(self._parse_header_line(splitData[8], 4, [10, 10, 10, 10, 10, 10, 10, 10],
                                                    [-1, -1, 2, 2, -1, -1, -1, -1],
                                                    ['', '', 'load_case', 'mode_n', '', '', '', '']))
                dset.update(self._parse_header_line(splitData[9], 4, [13, 13, 13, 13, 13, 13], [3, 3, 3, 3, 3, 3],
                                                    ['eig_r', 'eig_i', 'modal_a_r', 'modal_a_i', 'modal_b_r',
                                                     'modal_b_i']))
                dset.update({'modal_a': dset['modal_a_r'] + 1.j * dset['modal_a_i']})
                dset.update({'modal_b': dset['modal_b_r'] + 1.j * dset['modal_b_i']})
                dset.update({'eig': dset['eig_r'] + 1.j * dset['eig_i']})
                del dset['modal_a_r'], dset['modal_a_i'], dset['modal_b_r'], dset['modal_b_i']
                del dset['eig_r'], dset['eig_i']
            elif dset['analysis_type'] == 5:
                # frequency response
                dset.update(self._parse_header_line(splitData[8], 4, [10, 10, 10, 10, 10, 10, 10, 10],
                                                    [-1, -1, 2, 2, -1, -1, -1, -1],
                                                    ['', '', 'load_case', 'freq_step_n', '', '', '', '']))
                dset.update(self._parse_header_line(splitData[9], 1, [13, 13, 13, 13, 13, 13], [3, -1, -1, -1, -1, -1],
                                                    ['freq', '', '', '', '', '']))
                # Body
            splitData = ''.join(splitData[10:])
            values = np.asarray([float(str) for str in splitData.split()], 'd')
            if dset['data_type'] == 2:
                # real data
                if dset['n_data_per_node'] == 3:
                    dset['node_nums'] = values[:-3:4].copy()
                    dset['r1'] = values[1:-2:4].copy()
                    dset['r2'] = values[2:-1:4].copy()
                    dset['r3'] = values[3::4].copy()
                else:
                    dset['node_nums'] = values[:-6:7].copy()
                    dset['r1'] = values[1:-5:7].copy()
                    dset['r2'] = values[2:-4:7].copy()
                    dset['r3'] = values[3:-3:7].copy()
                    dset['r4'] = values[4:-2:7].copy()
                    dset['r5'] = values[5:-1:7].copy()
                    dset['r6'] = values[6::7].copy()
            elif dset['data_type'] == 5:
                # complex data
                if dset['n_data_per_node'] == 3:
                    dset['node_nums'] = values[:-6:7].copy()
                    dset['r1'] = values[1:-5:7] + 1.j * values[2:-4:7]
                    dset['r2'] = values[3:-3:7] + 1.j * values[4:-2:7]
                    dset['r3'] = values[5:-1:7] + 1.j * values[6::7]
                else:
                    raise UFFException('Cannot handle 6 points per node and complex data when reading data-set #55')
            else:
                raise UFFException('Error reading data-set #55')
        except:
            raise UFFException('Error reading data-set #55')
        del values
        return dset

    def _extract58(self, blockData):
        # Extract function at nodal DOF - data-set 58.
        dset = {'type': 58, 'binary': 0}
        try:
            binary = False
            split_header = b''.join(blockData.splitlines(True)[:13]).decode('utf-8',  errors='replace').splitlines(True)
            if len(split_header[1]) >= 7:
                if split_header[1][6].lower() == 'b':
                    # Read some addititional fields from the header section
                    binary = True
                    dset['binary'] = 1
                    dset.update(self._parse_header_line(split_header[1], 6, [6, 1, 6, 6, 12, 12, 6, 6, 12, 12],
                                                        [-1, -1, 2, 2, 2, 2, -1, -1, -1, -1],
                                                        ['', '', 'byte_ordering', 'fp_format', 'n_ascii_lines',
                                                         'n_bytes', '', '', '', '']))
            dset.update(self._parse_header_line(split_header[2], 1, [80], [1], ['id1']))
            dset.update(self._parse_header_line(split_header[3], 1, [80], [1], ['id2']))
            dset.update(self._parse_header_line(split_header[4], 1, [80], [1], ['id3']))  # usually for the date
            dset.update(self._parse_header_line(split_header[5], 1, [80], [1], ['id4']))
            dset.update(self._parse_header_line(split_header[6], 1, [80], [1], ['id5']))
            dset.update(self._parse_header_line(split_header[7], 1, [5, 10, 5, 10, 11, 10, 4, 11, 10, 4],
                                                [2, 2, 2, 2, 1, 2, 2, 1, 2, 2],
                                                ['func_type', 'func_id', 'ver_num', 'load_case_id', 'rsp_ent_name',
                                                 'rsp_node', 'rsp_dir', 'ref_ent_name',
                                                 'ref_node', 'ref_dir']))
            dset.update(self._parse_header_line(split_header[8], 6, [10, 10, 10, 13, 13, 13], [2, 2, 2, 3, 3, 3],
                                                ['ord_data_type', 'num_pts', 'abscissa_spacing', 'abscissa_min',
                                                 'abscissa_inc', 'z_axis_value']))
            dset.update(self._parse_header_line(split_header[9], 4, [10, 5, 5, 5, 21, 21], [2, 2, 2, 2, 1, 1],
                                                ['abscissa_spec_data_type', 'abscissa_len_unit_exp',
                                                 'abscissa_force_unit_exp', 'abscissa_temp_unit_exp',
                                                 'abscissa_axis_lab', 'abscissa_axis_units_lab']))
            dset.update(self._parse_header_line(split_header[10], 4, [10, 5, 5, 5, 21, 21], [2, 2, 2, 2, 1, 1],
                                                ['ordinate_spec_data_type', 'ordinate_len_unit_exp',
                                                 'ordinate_force_unit_exp', 'ordinate_temp_unit_exp',
                                                 'ordinate_axis_lab', 'ordinate_axis_units_lab']))
            dset.update(self._parse_header_line(split_header[11], 4, [10, 5, 5, 5, 21, 21], [2, 2, 2, 2, 1, 1],
                                                ['orddenom_spec_data_type', 'orddenom_len_unit_exp',
                                                 'orddenom_force_unit_exp', 'orddenom_temp_unit_exp',
                                                 'orddenom_axis_lab', 'orddenom_axis_units_lab']))
            dset.update(self._parse_header_line(split_header[12], 4, [10, 5, 5, 5, 21, 21], [2, 2, 2, 2, 1, 1],
                                                ['z_axis_spec_data_type', 'z_axis_len_unit_exp',
                                                 'z_axis_force_unit_exp', 'z_axis_temp_unit_exp', 'z_axis_axis_lab',
                                                 'z_axis_axis_units_lab']))
            # Body
            # splitData = ''.join(splitData[13:])
            if binary:
                split_data = b''.join(blockData.splitlines(True)[13:])
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
            else:
                values = []
                split_data = blockData.decode('utf-8', errors='replace').splitlines(True)[13:]
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
                    raise UFFException('Error reading data-set #58b; not proper data case.')

                values = np.asarray(values)
                # values = np.asarray([float(str) for str in splitData],'d')
            if (dset['ord_data_type'] == 2) or (dset['ord_data_type'] == 4):
                # Non-complex ordinate data
                if (dset['abscissa_spacing'] == 0):
                    # Uneven abscissa
                    dset['x'] = values[:-1:2].copy()
                    dset['data'] = values[1::2].copy()
                else:
                    # Even abscissa
                    nVal = len(values)
                    minVal = dset['abscissa_min']
                    d = dset['abscissa_inc']
                    dset['x'] = np.arange(minVal, minVal + nVal * d, d)
                    dset['data'] = values.copy()
            elif (dset['ord_data_type'] == 5) or (dset['ord_data_type'] == 6):
                # Complex ordinate data
                if (dset['abscissa_spacing'] == 0):
                    # Uneven abscissa
                    dset['x'] = values[:-2:3].copy()
                    dset['data'] = values[1:-1:3] + 1.j * values[2::3]
                else:
                    # Even abscissa
                    nVal = len(values) / 2
                    minVal = dset['abscissa_min']
                    d = dset['abscissa_inc']
                    dset['x'] = np.arange(minVal, minVal + nVal * d, d)
                    dset['data'] = values[0:-1:2] + 1.j * values[1::2]
            del values
        except:
            raise UFFException('Error reading data-set #58b')
        return dset

    def _opt_fields(self, dict, fieldsDict):
        # Sets the optional fields of the dict dictionary. Optionaly fields are
        # given in fieldsDict dictionary.
        for key in fieldsDict:
            #             if not dict.has_key(key):
            if not key in dict:
                dict.update({key: fieldsDict[key]})
        return dict

    def _parse_header_line(self, line, minValues, widths, types, names):
        # Parses the given line (a record in terms of UFF file) and returns all
        # the fields. line is a string representing the whole line.
        # width are fields widths to be read, types are field types
        #   1=string, 2=int, 3=float, -1=ignore the field
        # while names is a list of key (field) names.
        # Fields are split according to their widths as given in widths.
        # minValues specifies how many values (filds) are mandatory to read
        # from the line. Also, number of values found must not exceed the
        # number of fields requested by fieldsIn.
        # On the output, a dictionary of field names and corresponding
        # filed values is returned.
        fields = {}
        nFieldsReq = len(names)
        fieldsFromLine = []
        fieldsOut = {}

        # Extend the line if shorter than 80 chars
        ll = len(line)
        if ll < 80:
            line = line + ' ' * (80 - ll)
        # Parse the line for fields
        si = 0
        for n in range(0, len(widths)):
            fieldsFromLine.append(line[si:si + widths[n]].strip())
            si += widths[n]
        # Check for the number of fields,...
        nFields = len(fieldsFromLine)
        if (nFieldsReq < nFields) or (minValues > nFields):
            raise UFFException('Error parsing header section; too many or to less' + \
                               'fields found')
        # Mandatory fields
        for key, n in zip(names[:minValues], range(0, minValues)):
            if types[n] == -1:
                pass
            elif types[n] == 1:
                fieldsOut.update({key: fieldsFromLine[n]})
            elif types[n] == 2:
                fieldsOut.update({key: int(fieldsFromLine[n])})
            else:
                fieldsOut.update({key: float(fieldsFromLine[n])})
        # Optional fields
        for key, n in zip(names[minValues:nFields], range(minValues, nFields)):
            try:
                if types[n] == -1:
                    pass
                elif types[n] == 1:
                    fieldsOut.update({key: fieldsFromLine[n]})
                elif types[n] == 2:
                    fieldsOut.update({key: int(fieldsFromLine[n])})
                else:
                    fieldsOut.update({key: float(fieldsFromLine[n])})
            except ValueError:
                if types[n] == 1:
                    fieldsOut.update({key: ''})
                elif types[n] == 2:
                    fieldsOut.update({key: 0})
                else:
                    fieldsOut.update({key: 0.0})
        return fieldsOut


def prepare_test_15(save_to_file=''):
    dataset = {'type': 15,  # Nodes
               'node_nums': [16, 17, 18, 19, 20],  # I10, node label
               'def_cs': [11, 11, 11, 12, 12],  # I10, definition coordinate system number
               'disp_cs': [16, 16, 17, 18, 19],  # I10, displacement coordinate system number
               'color': [1, 3, 4, 5, 6],  # I10, color
               'x': [0.0, 1.53, 0.0, 1.53, 0.0],  # E13.5
               'y': [0.0, 0.0, 3.84, 3.84, 0.0],  # E13.5
               'z': [0.0, 0.0, 0.0, 0.0, 1.83]}  # E13.5
    dataset_out = dataset.copy()

    if save_to_file:
        if os.path.exists(save_to_file):
            os.remove(save_to_file)
        uffwrite = UFF(save_to_file)
        uffwrite._write_set(dataset, 'add')

    return dataset_out

def prepare_test_55(save_to_file=''):
    if save_to_file:
        if os.path.exists(save_to_file):
            os.remove(save_to_file)
    uff_datasets = []
    modes = [1, 2, 3]
    node_nums = [1, 2, 3, 4]
    freqs = [10.0, 12.0, 13.0]
    for i, b in enumerate(modes):
        mode_shape = np.random.normal(size=len(node_nums))
        name = 'TestCase'
        data = {
            'type': 55,
            'model_type': 1,
            'id1': 'NONE',
            'id2': 'NONE',
            'id3': 'NONE',
            'id4': 'NONE',
            'id5': 'NONE',
            'analysis_type': 2,
            'data_ch': 2,
            'spec_data_type': 8,
            'data_type': 2,
            'data_ch': 2,
            'r1': mode_shape,
            'r2': mode_shape,
            'r3': mode_shape,
            'n_data_per_node': 3,
            'node_nums': [1, 2, 3, 4],
            'load_case': 1,
            'mode_n': i + 1,
            'modal_m': 0,
            'freq': freqs[i],
            'modal_damp_vis': 0,
            'modal_damp_his': 0,
        }

        uff_datasets.append(data.copy())
        if save_to_file:
            uffwrite = UFF(save_to_file)
            uffwrite._write_set(data, 'add')
    return uff_datasets


def prepare_test_58(save_to_file=''):
    if save_to_file:
        if os.path.exists(save_to_file):
            os.remove(save_to_file)

    uff_datasets = []
    binary = [0, 1, 0]  # ascii of binary
    frequency = np.arange(10)
    np.random.seed(0)
    for i, b in enumerate(binary):
        print('Adding point {}'.format(i + 1))
        response_node = 1
        response_direction = 1
        reference_node = i + 1
        reference_direction = 1
        # this is an artificial 'frf'
        acceleration_complex = np.random.normal(size=len(frequency)) + \
                               1j * np.random.normal(size=len(frequency))
        name = 'TestCase'
        data = {'type': 58,
                'binary': binary[i],
                'func_type': 4,
                'rsp_node': response_node,
                'rsp_dir': response_direction,
                'ref_dir': reference_direction,
                'ref_node': reference_node,
                'data': acceleration_complex,
                'x': frequency,
                'id1': 'id1',
                'rsp_ent_name': name,
                'ref_ent_name': name,
                'abscissa_spacing': 1,
                'abscissa_spec_data_type': 18,
                'ordinate_spec_data_type': 12,
                'orddenom_spec_data_type': 13}
        uff_datasets.append(data.copy())
        if save_to_file:
            uffwrite = UFF(save_to_file)
            uffwrite._write_set(data, 'add')
    return uff_datasets


def prepare_test_82(save_to_file=''):
    dataset = {'type': 82,  # Tracelines
               'trace_num': 2,  # I10, trace line number
               'n_nodes': 7,  # I10, number of nodes defining trace line (max 250)
               'color': 30,  # I10, color
               'id': 'Identification line',  # 80A1, Identification line
               'nodes': np.array([0, 10, 13, 14, 15, 16, 17]),
               # I10, nodes defining trace line:
               #  > 0 draw line to node
               #  = 0 move to node
               }
    dataset_out = dataset.copy()

    if save_to_file:
        if os.path.exists(save_to_file):
            os.remove(save_to_file)
        uffwrite = UFF(save_to_file)
        uffwrite._write_set(dataset, 'add')

    return dataset_out


def prepare_test_151(save_to_file=''):
    dataset = {'type': 151,  # Header
               'model_name': 'Model file name',  # 80A1, model file name
               'description': 'Model file description',  # 80A1, model file description
               'db_app': 'Program which created DB',  # 80A1, program which created DB
               'date_db_created': '27-Jan-16',  # 10A1, date database created (DD-MMM-YY)
               'time_db_created': '14:38:15',  # 10A1, time database created (HH:MM:SS)
               'version_db1': 1,  # I10, Version from database
               'version_db2': 2,  # I10, Subversion from database
               'file_type': 0,  # I10, File type (0  Universal, 1 Archive, 2 Other)
               'date_db_saved': '28-Jan-16',  # 10A1, date database saved (DD-MMM-YY)
               'time_db_saved': '14:38:16',  # 10A1, time database saved (HH:MM:SS)
               'program': 'OpenModal',  # 80A1, program which created DB
               'date_db_written': '29-Jan-16',  # 10A1, date database written (DD-MMM-YY)
               'time_db_written': '14:38:17',  # 10A1, time database written (HH:MM:SS)
               }

    dataset_out = dataset.copy()

    if save_to_file:
        if os.path.exists(save_to_file):
            os.remove(save_to_file)
        uffwrite = UFF(save_to_file)
        uffwrite._write_set(dataset, 'add')

    return dataset_out


def prepare_test_164(save_to_file=''):
    dataset = {'type': 164,  # Universal Dataset
               'units_code': 1,  # I10, units code
               'units_description': 'SI units',  # 20A1, units description
               'temp_mode': 1,  # I10, temperature mode
               # Unit factors
               # for converting universal file units to SI.
               # To convert from universal file units to SI divide by
               # the appropriate factor listed below.
               'length': 3.28083989501312334,  # D25.17, length
               'force': 2.24808943099710480e-01,  # D25.17, force
               'temp': 1.8,  # D25.17, temperature
               'temp_offset': 459.67,  # D25.17, temperature offset
               }
    dataset_out = dataset.copy()

    if save_to_file:
        if os.path.exists(save_to_file):
            os.remove(save_to_file)
        uffwrite = UFF(save_to_file)
        uffwrite._write_set(dataset, 'add')

    return dataset_out


if __name__ == '__main__':
    #uff_ascii = UFF('./data/beam.uff')
    #a = uff_ascii.read_sets(0)
    #print(a)
    #prepare_test_55('./data/test_uff55.uff')
    # uff_ascii = UFF('./data/Artemis export - Geometry RPBC_setup_05_14102016_105117.uff')
    #uff_ascii = UFF('./data/no_spacing2_UFF58_ascii.uff')
    #uff_ascii = UFF('./data/mesh_Oros-modal_uff15_uff2412.unv')
    uff_ascii = UFF('./data/DS2414_disp_file.uff')
    a = uff_ascii.read_sets(3)
    for _ in a.keys():
        if _ != 'data':
            print(_, ':', a[_])
    #print(sum(a['data']))