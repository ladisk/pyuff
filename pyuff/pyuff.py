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
import numpy as np


from .tools import UFFException

from .datasets.dataset_15 import _write15, _extract15
from .datasets.dataset_18 import _extract18
from .datasets.dataset_55 import _write55, _extract55
from .datasets.dataset_58 import _write58, _extract58
from .datasets.dataset_82 import _write82, _extract82
from .datasets.dataset_151 import _write151, _extract151
from .datasets.dataset_164 import _write164, _extract164
from .datasets.dataset_2411 import _write2411, _extract2411
from .datasets.dataset_2412 import _write2412, _extract2412
from .datasets.dataset_2414 import _write2414, _extract2414
from .datasets.dataset_2420 import _write2420, _extract2420

__version__ = '1.26'
_SUPPORTED_SETS = ['151', '15', '55', '58', '58b', '82', '164', '2411', '2412', '2414', '2420']


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
                self._setTypes = np.zeros(nBlocks).astype(int)
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
        """
        Reads n-th set from UFF file. 
        n can be an integer between 0 and nSets-1. 
        User must be sure that, since the last reading/writing/refreshing, 
        the data has not changed by some other means than through the
        UFF object. The method returns dset dictionary.
        """
        
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
            dset = _extract15(blockData)
        elif self._setTypes[int(n)] == 18:
            dset = _extract18(blockData)  # TEMP ADD
        elif self._setTypes[int(n)] == 55:
            dset = _extract55(blockData)
        elif self._setTypes[int(n)] == 58:
            dset = _extract58(blockData)
        elif self._setTypes[int(n)] == 82:
            dset = _extract82(blockData)
        elif self._setTypes[int(n)] == 151:
            dset = _extract151(blockData)
        elif self._setTypes[int(n)] == 164:
            dset = _extract164(blockData)
        elif self._setTypes[int(n)] == 2411:
            dset = _extract2411(blockData)  # TEMP ADD
        elif self._setTypes[int(n)] == 2412:
            dset = _extract2412(blockData)
        elif self._setTypes[int(n)] == 2414:
            dset = _extract2414(blockData) 
        elif self._setTypes[int(n)] == 2420:
            dset = _extract2420(blockData)
        else:
            dset['type'] = self._setTypes[int(n)]
            # Unsupported data-set - do nothing
            pass
        return dset

    def _write_set(self, dset, mode='add'):
        """
        Writes UFF data (UFF data-sets) to the file.  The mode can be
        either 'add' (default) or 'overwrite'. The dset is a
        dictionary of keys and corresponding values. Unsupported
        data-set will be ignored.
         
        For each data-set, there are some optional and some required fields at
        dset dictionary. Also, in general, the sum of the required
        and the optional fields together can be less then the number of fields
        read from the same type of data-set; the reason is that for some
        data-sets some fields are set automatically. Optional fields are
        calculated automatically and the dset is updated - as dset is actually
        an alias (aka pointer), this is reflected at the caller too.
        
        """
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
                _write15(fh, dset)
            elif setType == 55:
                _write55(fh, dset)
            elif setType == 58:
                _write58(fh, dset, mode, _fileName=self._fileName)
            elif setType == 82:
                _write82(fh, dset)
            elif setType == 151:
                _write151(fh, dset)
            elif setType == 164:
                _write164(fh, dset)
            elif setType == 2411:
                _write2411(fh, dset)
            elif setType == 2412:
                _write2412(fh, dset)
            elif setType == 2414:
                _write2414(fh, dset)
            elif setType == 2420:
                _write2420(fh, dset)
            else:
                # Unsupported data-set - do nothing
                pass
        except:
            fh.close()
            raise  # re-raise the last exception
        else:
            fh.close()
        self.refresh()


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