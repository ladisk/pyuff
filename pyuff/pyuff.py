"""
==========
pyuff module
==========

This module defines an UFF class to manipulate with the
UFF (Universal File Format) files, i.e., to read from and write
to UFF files. Among the variety of UFF formats, only some of the
formats (data-set types) frequently used in structural dynamics
are supported: **15, 55, 58, 58b, 82, 151, 164, 2411, 2412, 2414, 2420, 2429, 2467** 
Data-set **58b** is actually a hybrid format [1]_ where the signal is written in the
binary form, while the header-part is slightly different from 58 but still in the
ascii format.

An UFF file is a file that can have many data-sets of either ascii or binary
data where data-set is a block of data between the start and end tags ``____-1``
(``_`` representing the space character). Refer to [1]_ and [2]_ for
more information about the UFF format.

Sources:
    .. [1] https://www.ceas3.uc.edu/sdrluff/
    .. [2] Matlab's ``readuff`` and ``writeuff`` functions:
       http://www.mathworks.com/matlabcentral/fileexchange/loadFile.do?objectId=6395

Acknowledgement:
    * This source (py2.7) was first written in 2007, 2008 by Primoz Cermelj (primoz.cermelj@gmail.com)
    * As part of the www.openmodal.com project the first source was adopted for Python 3 by
      Matjaz Mrsnik  <matjaz.mrsnik@gmail.com>
    * 2014-2017 the package was part of the OpenModal project maintained by: Martin Česnik, 
      Matjaž Mršnik, Miha Pirnat, Janko Slavič, Blaž Starc (in alphabetic order)
    * The package is maintained by Janko Slavič <janko.slavic@fs.uni-lj.si>

Notes:
    * by default 58 data-set is written in double precision (see option `force_double=True`),
      even if it is read in single precision ().
      
    Example:
    >>> import pyuff
    >>> uff_file = pyuff.UFF('beam.uff')
    >>> uff_file.file_exists()
    True
"""
import os
import numpy as np
import warnings
warnings.simplefilter("default")


from .datasets.dataset_15 import _write15, _extract15, get_structure_15
from .datasets.dataset_18 import _extract18, get_structure_18
from .datasets.dataset_55 import _write55, _extract55, get_structure_55
from .datasets.dataset_58 import _write58, _extract58, get_structure_58, fix_58b
from .datasets.dataset_82 import _write82, _extract82, get_structure_82
from .datasets.dataset_151 import _write151, _extract151, get_structure_151
from .datasets.dataset_164 import _write164, _extract164, get_structure_164
from .datasets.dataset_2411 import _write2411, _extract2411, get_structure_2411
from .datasets.dataset_2412 import _write2412, _extract2412, get_structure_2412
from .datasets.dataset_2414 import _write2414, _extract2414, get_structure_2414
from .datasets.dataset_2420 import _write2420, _extract2420, get_structure_2420
from .datasets.dataset_2429 import _write2429, _extract2429, get_structure_2429
from .datasets.dataset_2467 import _write2467, _extract2467, get_structure_2467

_SUPPORTED_SETS = ['15', '55', '58', '58b', '82', '151','164', '2411', '2412', '2414', '2420', '2429', '2467']


class UFF:
    """
    Manages data reading and writing from/to the UFF file.
    
    The UFF class instance requires exactly 1 parameter - a file name of a
    universal file. If the file does not exist, no basic file info will be
    extracted and the status will be False - indicating that the file is not
    refreshed. Hovewer, when one tries to read one or more data-sets, the file
    must exist or the Exception will be raised.
    
    The file, given as a parameter to the UFF instance, is open only when
    reading from or writing to the file. The UFF instance refreshes the file
    automatically - use ``UFF.get_status()`` to see the refresh status); note
    that this works fine if the file is being changed only through the UFF
    instance and not by other functions or even by other means, e.g.,
    externally. If the file is changed externally, the ``UFF.refresh()`` should
    be invoked before any reading or writing.
    
    All array-type data are read/written using numpy's ``np.array`` module.
    """

    def __init__(self, filename=None, fileName=None):
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
        if filename != None:
            self._filename = filename
        
        elif fileName != None:
            self._filename = fileName
            warnings.warn('Argument ``fileName`` will be deprecated in the future. Please use ``filename``')
        

        
        self._block_ind = []  # an array of block indices: start-end pairs in rows
        self._refreshed = False
        self._n_sets = 0  # number of sets found in file
        self._set_types = np.array(())  # list of set-type numbers
        self._set_formats = np.array(())  # list of set-format numbers (0=ascii,1=binary)
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
        return self._n_sets

    def get_set_types(self):
        """
        Returns an array of data-set types. All valid data-sets are returned,
        even those that are not supported, i.e., whose contents will not be
        read.
        """
        if not self._refreshed:
            self.refresh()
        return self._set_types

    def get_set_formats(self):
        """Returns an array of data-set formats: 0=ascii, 1=binary."""
        if not self._refreshed:
            self.refresh()
        return self._set_formats

    def get_file_name(self):
        """Returns the file name (as a string) associated with the uff object."""
        return self._filename

    def file_exists(self):
        """
        Returns true if the file exists and False otherwise. If the file does
        not exist, invoking one of the read methods would raise the Exception
        exception.
        """
        return os.path.exists(self._filename)

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
            fh = open(self._filename, 'rb')
        #             fh = open(self._filename, 'rt')
        except:
            raise Exception('Cannot access the file %s' % self._filename)
        else:
            try:
                # Parses the entire file for '    -1' tags and extracts
                # the corresponding indices
                data = fh.read()
                data_len = len(data)
                ind = -1
                block_ind = []
                while True:
                    ind = data.find(b'    -1', ind + 1)
                    if ind == -1:
                        break
                    if ind+6 == data_len:
                        block_ind.append(ind)
                    # Check if the block is valid and ends with LF or CR
                    elif ind+6 < data_len and data[ind+6] in [10, 13]:
                        block_ind.append(ind)
                    # Check the line continues with blanks up to character 80
                    elif ind+80 < data_len and data[ind+6:ind+80] == \
                        b'                                                                          ':
                        block_ind.append(ind)
                    
                block_ind = np.asarray(block_ind, dtype='int64')

                # Constructs block indices of start and end values; each pair
                # points to start and end offset of the data-set (block) data,
                # but, the start '    -1' tag is included while the end one is
                # excluded.
                n_blocks = int(np.floor(len(block_ind) / 2.0))
                if n_blocks == 0:
                    # No valid blocks found but the file is still considered
                    # being refreshed
                    fh.close()
                    self._refreshed = True
                    return self._refreshed
                self._block_ind = np.zeros((n_blocks, 2), dtype='int64')
                self._block_ind[:, 0] = block_ind[:-1:2].copy()
                self._block_ind[:, 1] = block_ind[1::2].copy() - 1

                # Go through all the data-sets (blocks) and extract data-set
                # type and the property whether the data-set is in binary
                # or ascii format
                self._n_sets = n_blocks
                self._set_types = np.zeros(n_blocks).astype(int)
                self._set_formats = np.zeros(n_blocks)
                for ii in range(0, self._n_sets):
                    si = self._block_ind[ii, 0]
                    ei = self._block_ind[ii, 1]
                    try:
                        block_data = data[si:ei + 1].splitlines()
                        self._set_types[ii] = int(block_data[1][0:6])
                        if block_data[1][6].lower() == 'b':
                            self._set_formats[ii] = 1
                    except:
                        # Some non-valid blocks found; ignore the exception
                        pass
                del block_ind
            except:
                fh.close()
                raise Exception('Error refreshing UFF file: ' + self._filename)
            else:
                self._refreshed = True
                fh.close()
                return self._refreshed

    def read_sets(self, setn=None, header_only=False):
        """
        Reads sets.
        
        The method returns a list of dset dictionaries - as many dictionaries as there are sets. 
        Unknown data-sets are returned empty.
        User must be sure that, since the last reading/writing/refreshing,
        the data has not changed by some other means than through the
        UFF object.
        
        :param setn: None(default), all sets are read in None (default). 
                     If a number is given, then only a particular set is read. 
        :param header_only: False (default), if True header is read, only
                     This usefull for large files.         
        """
        dset = []
        if setn is None:
            read_range = range(0, self._n_sets)
        else:
            if (not type(setn).__name__ == 'list'):
                read_range = [setn]
            else:
                read_range = setn
        if not self.file_exists():
            raise Exception('Cannot read from a non-existing file: ' + self._filename)
        if not self._refreshed:
            if not self.refresh():
                raise Exception('Cannot read from the file: ' + self._filename)
        try:
            for ii in read_range:
                dset.append(self._read_set(ii, header_only=header_only))
        except Exception as msg:
            if hasattr(msg, 'value'):
                raise Exception('Error when reading ' + str(ii) + '-th data-set: ' + msg.value)
            else:
                raise Exception('Error when reading data-set(s).')
        if len(dset) == 1:
            dset = dset[0]
        return dset

    def write_sets(self, dsets, mode='add', force_double=True):
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

        The required fields are:
            dset - dictionary representing the data-set
            mode - 'add' or 'overwrite'
            force_double - True or False (default True). Single precision should be avoided, 
                           therefore by default data is saved with double precision.
        """
        if (not type(dsets).__name__ == 'list'):
            dsets = [dsets]
        n_sets = len(dsets)
        if n_sets < 1:
            raise Exception('Nothing to write')
        if mode.lower() == 'overwrite':
            # overwrite mode; first set is written in the overwrite mode, others
            # in add mode
            self._write_set(dsets[0], 'overwrite', force_double=force_double)
            for ii in range(1, n_sets):
                self._write_set(dsets[ii], 'add', force_double=force_double)
        elif mode.lower() == 'add':
            # add mode; all the sets are written in the add mode
            for ii in range(0, n_sets):
                self._write_set(dsets[ii], 'add', force_double=force_double)
        else:
            raise Exception('Unknown mode: ' + mode)

    def _read_set(self, n, header_only=False):
        """
        Reads n-th set from UFF file. 

        n can be an integer between 0 and n_sets-1. 
        
        User must be sure that, since the last reading/writing/refreshing, 
        the data has not changed by some other means than through the
        UFF object. The method returns dset dictionary.

        :param header_only: False (default), if True header is read, only
                This usefull for large files.    
        """
        
        dset = {}
        if not self.file_exists():
            raise Exception('Cannot read from a non-existing file: ' + self._filename)
        if not self._refreshed:
            if not self.refresh():
                raise Exception('Cannot read from the file: ' + self._filename + '. The file cannot be refreshed.')
        if (n > self._n_sets - 1) or (n < 0):
            raise Exception('Cannot read data-set: ' + str(int(n)) + \
                               '. Data-set number to high or to low.')
        # Read n-th data-set data (one block)
        try:
            fh = open(self._filename, 'rb')
        except:
            raise Exception('Cannot access the file: ' + self._filename + ' to read from.')
        else:
            try:
                si = self._block_ind[n][0]  # start offset
                ei = self._block_ind[n][1]  # end offset
                fh.seek(si)
                if self._set_types[int(n)] == 58:
                    block_data = fh.read(ei - si + 1)  # decoding is handled later in _extract58
                else:
                    block_data = fh.read(ei - si + 1).decode('utf-8', errors='replace')
            except:
                fh.close()
                raise Exception('Error reading data-set #: ' + int(n))
            else:
                fh.close()
        # Extracts the dset
        if self._set_types[int(n)] == 15:
            dset = _extract15(block_data)
        elif self._set_types[int(n)] == 18:
            dset = _extract18(block_data)  # TEMP ADD
        elif self._set_types[int(n)] == 55:
            dset = _extract55(block_data)
        elif self._set_types[int(n)] == 58:
            dset = _extract58(block_data, header_only=header_only)
        elif self._set_types[int(n)] == 82:
            dset = _extract82(block_data)
        elif self._set_types[int(n)] == 151:
            dset = _extract151(block_data)
        elif self._set_types[int(n)] == 164:
            dset = _extract164(block_data)
        elif self._set_types[int(n)] == 2411:
            dset = _extract2411(block_data)  # TEMP ADD
        elif self._set_types[int(n)] == 2412:
            dset = _extract2412(block_data)
        elif self._set_types[int(n)] == 2414:
            dset = _extract2414(block_data) 
        elif self._set_types[int(n)] == 2420:
            dset = _extract2420(block_data)
        elif self._set_types[int(n)] == 2429:
            dset = _extract2429(block_data)
        elif self._set_types[int(n)] == 2467:
            dset = _extract2467(block_data)
        else:
            dset['type'] = self._set_types[int(n)]
            # Unsupported data-set - do nothing
            pass
        return dset

    def _write_set(self, dset, mode='add', force_double=True):
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
                fh = open(self._filename, 'wt')
            except:
                raise Exception('Cannot access the file: ' + self._filename + ' to write to.')
        elif mode.lower() == 'add':
            # add (append) mode
            try:
                fh = open(self._filename, 'at')
            except:
                raise Exception('Cannot access the file: ' + self._filename + ' to write to.')
        else:
            raise Exception('Unknown mode: ' + mode)
        try:
            # Actual writing
            try:
                set_type = dset['type']
            except:
                fh.close()
                raise Exception('Data-set\'s dictionary is missing the required \'type\' key')
            # handle nan or inf
            if 'data' in dset.keys():
                dset['data'] = np.nan_to_num(dset['data'])

            if set_type == 15:
                _write15(fh, dset)
            elif set_type == 55:
                _write55(fh, dset)
            elif set_type == 58:
                _write58(fh, dset, mode, _filename=self._filename, force_double=force_double)
            elif set_type == 82:
                _write82(fh, dset)
            elif set_type == 151:
                _write151(fh, dset)
            elif set_type == 164:
                _write164(fh, dset)
            elif set_type == 2411:
                _write2411(fh, dset)
            elif set_type == 2412:
                _write2412(fh, dset)
            elif set_type == 2414:
                _write2414(fh, dset)
            elif set_type == 2420:
                _write2420(fh, dset)
            elif set_type == 2429:
                _write2429(fh, dset)
            elif set_type == 2467:
                _write2467(fh, dset)
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
    pass