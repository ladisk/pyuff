"""
==========
pyuff module
==========

This module defines an UFF class to manipulate with the
UFF (Universal File Format) files, i.e., to read from and write
to UFF files. Among the variety of UFF formats, only some of the
formats (data-set types) frequently used in structural dynamics
are supported: **15, 18, 55, 58, 58b, 82, 151, 164, 1858, 2400, 2411, 2412, 2414, 2420, 2429, 2467** 
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
    * by default 58 data-set is written in double precision (see option ``force_double=True``),
      even if it is read in single precision.

Example:
    >>> import pyuff
    >>> uff_file = pyuff.UFF('data/beam.uff')
    >>> uff_file.file_exists()
    True
"""
import os
import numpy as np
import warnings


from .datasets.dataset_15 import _write15, _extract15, get_structure_15
from .datasets.dataset_18 import _write18, _extract18, get_structure_18
from .datasets.dataset_55 import _write55, _extract55, get_structure_55
from .datasets.dataset_58 import _write58, _extract58, get_structure_58, fix_58b
from .datasets.dataset_82 import _write82, _extract82, get_structure_82
from .datasets.dataset_151 import _write151, _extract151, get_structure_151
from .datasets.dataset_164 import _write164, _extract164, get_structure_164
from .datasets.dataset_1858 import _write1858, _extract1858, get_structure_1858
from .datasets.dataset_2400 import _write2400, _extract2400, get_structure_2400
from .datasets.dataset_2411 import _write2411, _extract2411, get_structure_2411
from .datasets.dataset_2412 import _write2412, _extract2412, get_structure_2412
from .datasets.dataset_2414 import _write2414, _extract2414, get_structure_2414
from .datasets.dataset_2420 import _write2420, _extract2420, get_structure_2420
from .datasets.dataset_2429 import _write2429, _extract2429, get_structure_2429
from .datasets.dataset_2467 import _write2467, _extract2467, get_structure_2467

_WRITERS = {15: _write15, 18: _write18, 55: _write55, 58: _write58, 82: _write82, 151: _write151,
            164: _write164, 1858: _write1858, 2400: _write2400, 2411: _write2411, 2412: _write2412,
            2414: _write2414, 2420: _write2420, 2429: _write2429, 2467: _write2467}

_EXTRACTORS = {15: _extract15, 18: _extract18, 55: _extract55, 82: _extract82, 151: _extract151,
               164: _extract164, 1858: _extract1858, 2400: _extract2400, 2411: _extract2411,
               2412: _extract2412, 2414: _extract2414, 2420: _extract2420, 2429: _extract2429,
               2467: _extract2467}

_SUPPORTED_SETS = ['15', '18', '55', '58', '58b', '82', '151', '164', '1858', '2400', '2411', '2412', '2414', '2420', '2429', '2467']


_BLANKS_74 = b' ' * 74


def _is_set_number(line):
    """True if the line gives a data-set number: I6, optionally followed by 'b' (58b)."""
    return line[:6].strip(b' ').isdigit() and line[6:7] in (b'', b' ', b'b', b'B')


def _find_block_tags(data):
    """Returns the offsets of the '    -1' tags that start and end the data-sets.

    By the specification, a tag is a line '    -1' (a minus sign in column 5, a 1 in
    column 6, the rest of the line blank), and a start tag is followed by a line with
    the data-set number. The only exception is the end tag of a binary data-set (58b):
    it immediately follows the binary data, so it can be anywhere in a line. A '-1' at
    the end of a record (e.g. an I10 field) is a value, not a tag.
    """
    data_len = len(data)
    tags = []
    in_block = False
    binary = False
    ind = -1
    while True:
        ind = data.find(b'    -1', ind + 1)
        if ind == -1:
            break
        end = ind + 6
        if not (end == data_len or data[end] in (10, 13) or
                (ind + 80 < data_len and data[end:ind + 80] == _BLANKS_74)):
            continue
        if not (in_block and binary):
            # the tag begins the line (the search for CR is limited to the line)
            last_lf = data.rfind(b'\n', 0, ind)
            line_start = max(last_lf, data.rfind(b'\r', last_lf + 1, ind)) + 1
            if ind != line_start:
                continue
        if in_block:
            in_block = False
        else:
            # the line after a start tag gives the data-set number ('b' after it marks
            # a binary data-set); as in refresh(), it is within 200 bytes of the tag
            lines = data[ind:ind + 200].splitlines()
            if len(lines) < 2 or not _is_set_number(lines[1]):
                continue
            in_block = True
            binary = lines[1][6:7].lower() == b'b'
        tags.append(ind)
    return tags


class UFF:
    """
    A universal file (UFF/UNV), for reading and writing its data-sets.

    A universal file is a text file of data-sets, each between two lines with ``-1``. Each
    data-set is read into a dictionary: its ``type`` key is the data-set number (e.g. 58)
    and the other keys are the fields of the data-set. Numerical data are NumPy arrays.

    The file is opened only while reading or writing. When the object is created, the file
    (if it exists) is indexed: the number, types and formats of its data-sets are found.
    The index is updated after writing through this object; if the file is changed by
    other means, call :meth:`refresh` before reading.

    Parameters
    ----------
    filename : str or path-like
        Name of the file. It does not need to exist for writing.
    fileName : str, optional
        Deprecated alias of ``filename``.

    Examples
    --------
    >>> import pyuff
    >>> uff_file = pyuff.UFF('data/beam.uff')
    >>> uff_file.get_n_sets()
    7
    >>> data = uff_file.read_sets()
    >>> data[0]['type']
    151
    """

    def __init__(self, filename=None, fileName=None):
        """Index the file, if it exists (see :meth:`refresh`)."""
        # Some "private" members
        if filename != None:
            self._filename = filename
        
        elif fileName != None:
            self._filename = fileName
            warnings.warn('Argument ``fileName`` will be deprecated in the future. Please use ``filename``',
                          FutureWarning, stacklevel=2)
        

        
        self._block_ind = []  # an array of block indices: start-end pairs in rows
        self._refreshed = False
        self._n_sets = 0  # number of sets found in file
        self._set_types = np.array(())  # list of set-type numbers
        self._set_formats = np.array(())  # list of set-format numbers (0=ascii,1=binary)
        # Refresh
        self.refresh()

    def get_supported_sets(self):
        """
        Return the data-set types that can be read and written.

        Returns
        -------
        list of str
            Data-set types, e.g. ``'58'``; ``'58b'`` is the binary form of data-set 58.
        """
        return _SUPPORTED_SETS

    def get_n_sets(self):
        """
        Return the number of data-sets in the file.

        Returns
        -------
        int
            Number of data-sets, including those of types that are not supported.
        """
        if not self._refreshed:
            self.refresh()
        return self._n_sets

    def get_set_types(self):
        """
        Return the type of each data-set in the file.

        Returns
        -------
        numpy.ndarray
            Data-set types (e.g. 151, 164, 58), in the order of the data-sets in the file,
            including the types that are not supported.
        """
        if not self._refreshed:
            self.refresh()
        return self._set_types

    def get_set_formats(self):
        """
        Return the format of each data-set in the file.

        Returns
        -------
        numpy.ndarray
            0 for an ASCII data-set and 1 for a binary data-set (58b), in the order of the
            data-sets in the file.
        """
        if not self._refreshed:
            self.refresh()
        return self._set_formats

    def get_file_name(self):
        """Return the name of the file, as given when the object was created."""
        return self._filename

    def file_exists(self):
        """
        Return whether the file exists.

        Returns
        -------
        bool
            True if the file exists; reading a file that does not exist raises an exception.
        """
        return os.path.exists(self._filename)

    def get_status(self):
        """
        Return whether the index of the file is up to date.

        Returns
        -------
        bool
            True if the file has been indexed since it was last written through this
            object.
        """
        return self._refreshed

    def refresh(self):
        """
        Index the file again: find the number, types and formats of its data-sets.

        Call this method when the file was changed by other means than through this object.

        Returns
        -------
        bool
            True if the file was indexed, False if it does not exist.
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
                block_ind = np.asarray(_find_block_tags(data), dtype='int64')

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
                        # only the second line (data-set type, 'b' for binary) is
                        # needed; it follows the '    -1' line within 200 bytes
                        type_line = data[si:min(si + 200, ei + 1)].splitlines()[1]
                        self._set_types[ii] = int(type_line[0:6])
                        if type_line[6:7].lower() == b'b':
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
        Read data-sets from the file.

        Parameters
        ----------
        setn : int or list of int, optional
            Index of the data-set to read (0 for the first one), or a list of indices.
            If None (default), all data-sets are read.
        header_only : bool, optional
            If True, data-sets 58 are read without their data values (``x`` and
            ``data``), which is faster for large files. Default: False.

        Returns
        -------
        dict or list of dict
            One dictionary for each data-set read; a single dictionary if one data-set
            is read. Data-sets of unsupported types give only ``{'type': N}``.

        Examples
        --------
        >>> import pyuff
        >>> uff_file = pyuff.UFF('data/beam.uff')
        >>> header = uff_file.read_sets(0)
        >>> frfs = uff_file.read_sets([4, 5, 6])
        >>> len(frfs)
        3
        """
        dset = []
        if not self.file_exists():
            raise Exception('Cannot read from a non-existing file: ' + self._filename)
        if not self._refreshed:
            if not self.refresh():
                raise Exception('Cannot read from the file: ' + self._filename)
        if setn is None:
            read_range = range(0, self._n_sets)
        else:
            if (not type(setn).__name__ == 'list'):
                read_range = [setn]
            else:
                read_range = setn
        try:
            # one file handle for all data-sets read
            with open(self._filename, 'rb') as fh:
                for ii in read_range:
                    dset.append(self._read_set(ii, header_only=header_only, fh=fh))
        except Exception as msg:
            if hasattr(msg, 'value'):
                raise Exception('Error when reading ' + str(ii) + '-th data-set: ' + msg.value)
            else:
                raise Exception('Error when reading data-set(s).') from msg
        if len(dset) == 1:
            dset = dset[0]
        return dset

    def write_sets(self, dsets, mode='add', force_double=True):
        """
        Write data-sets to the file.

        The data-sets are dictionaries, as returned by the ``prepare_*`` functions (e.g.
        :func:`pyuff.prepare_58`) or by :meth:`read_sets`. Optional fields that are missing
        or None get their default values; the dictionaries are updated with them.

        Text longer than its field is truncated, with a warning. NaN and infinite values
        in ``data`` are written as 0 and as the largest finite float, with a warning.

        Parameters
        ----------
        dsets : dict or list of dict
            The data-sets to write.
        mode : {'add', 'overwrite'}, optional
            ``'add'`` (default) appends the data-sets to the file; ``'overwrite'`` starts a
            new file.
        force_double : bool, optional
            If True (default), data-set 58 is written in double precision, also if it was
            read in single precision. If False, the precision is given by
            ``ord_data_type`` of each data-set (2 and 5 single, 4 and 6 double).

        Examples
        --------
        >>> import numpy as np
        >>> import pyuff
        >>> frf = pyuff.prepare_58(
        ...     func_type=4, rsp_node=1, rsp_dir=3, ref_node=1, ref_dir=3,
        ...     data=np.ones(5, dtype=complex), x=np.arange(5.0), abscissa_spacing=1,
        ...     abscissa_spec_data_type=18, ordinate_spec_data_type=12,
        ...     orddenom_spec_data_type=13)
        >>> pyuff.UFF('frf.uff').write_sets([frf], mode='overwrite')
        """
        if (not type(dsets).__name__ == 'list'):
            dsets = [dsets]
        n_sets = len(dsets)
        if n_sets < 1:
            raise Exception('Nothing to write')
        if mode.lower() == 'overwrite':
            open_mode = 'wt'
        elif mode.lower() == 'add':
            open_mode = 'at'
        else:
            raise Exception('Unknown mode: ' + mode)
        # one file handle for all data-sets (opening/closing per data-set dominated
        # the time of writing many small data-sets)
        try:
            # newline='\n': the same file (LF line ends) on every platform
            fh = open(self._filename, open_mode, encoding='utf-8', newline='\n')
        except:
            raise Exception('Cannot access the file: ' + self._filename + ' to write to.')
        try:
            for ii in range(0, n_sets):
                # the first data-set of 'overwrite' starts the file, the others are added
                self._write_dset(fh, dsets[ii], mode.lower() if ii == 0 else 'add', force_double=force_double)
        finally:
            fh.close()
            self._refreshed = False
        self.refresh()

    def _read_set(self, n, header_only=False, fh=None):
        """
        Reads n-th set from UFF file. 

        n can be an integer between 0 and n_sets-1. 
        
        User must be sure that, since the last reading/writing/refreshing, 
        the data has not changed by some other means than through the
        UFF object. The method returns dset dictionary.

        :param header_only: False (default), if True header is read, only
                This usefull for large files.    
        :param fh: optional file handle (opened in 'rb' mode) of the file; if None,
                the file is opened and closed here.
        """
        if fh is None:
            if not self.file_exists():
                raise Exception('Cannot read from a non-existing file: ' + self._filename)
        if not self._refreshed:
            if not self.refresh():
                raise Exception('Cannot read from the file: ' + self._filename + '. The file cannot be refreshed.')
        if (n > self._n_sets - 1) or (n < 0):
            raise Exception('Cannot read data-set: ' + str(int(n)) +
                            '. Data-set number to high or to low.')
        set_type = int(self._set_types[int(n)])
        # Read n-th data-set data (one block)
        own_fh = fh is None
        if own_fh:
            try:
                fh = open(self._filename, 'rb')
            except:
                raise Exception('Cannot access the file: ' + self._filename + ' to read from.')
        try:
            si = self._block_ind[n][0]  # start offset
            ei = self._block_ind[n][1]  # end offset
            fh.seek(si)
            if set_type == 58:
                block_data = fh.read(ei - si + 1)  # decoding is handled later in _extract58
            else:
                block_data = fh.read(ei - si + 1).decode('utf-8', errors='replace')
        except:
            raise Exception('Error reading data-set #: ' + str(int(n)))
        finally:
            if own_fh:
                fh.close()
        # Extracts the dset
        if set_type == 58:
            return _extract58(block_data, header_only=header_only)
        extract = _EXTRACTORS.get(set_type)
        if extract is not None:
            return extract(block_data)
        # Unsupported data-set - do nothing
        return {'type': self._set_types[int(n)]}

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
                fh = open(self._filename, 'wt', encoding='utf-8', newline='\n')
            except:
                raise Exception('Cannot access the file: ' + self._filename + ' to write to.')
        elif mode.lower() == 'add':
            # add (append) mode
            try:
                fh = open(self._filename, 'at', encoding='utf-8', newline='\n')
            except:
                raise Exception('Cannot access the file: ' + self._filename + ' to write to.')
        else:
            raise Exception('Unknown mode: ' + mode)
        try:
            self._write_dset(fh, dset, mode, force_double=force_double)
        finally:
            fh.close()
        # the file has changed: its index of data-sets is rebuilt when it is
        # needed next, not after every data-set (write_sets writes many)
        self._refreshed = False

    def _write_dset(self, fh, dset, mode='add', force_double=True):
        """Writes one data-set to the open (text) file fh."""
        # Actual writing
        try:
            set_type = dset['type']
        except:
            raise Exception("Data-set's dictionary is missing the required 'type' key")
        # handle nan or inf
        if 'data' in dset.keys():
            data = np.asarray(dset['data'])
            if np.issubdtype(data.dtype, np.inexact):
                n_not_finite = np.count_nonzero(~np.isfinite(data))
                if n_not_finite:
                    warnings.warn('Data-set %s: %i of %i data values are NaN or infinite; '
                                  'NaN is written as 0 and infinity as the largest finite '
                                  'float.' % (set_type, n_not_finite, data.size), stacklevel=3)
            dset['data'] = np.nan_to_num(dset['data'])
        writer = _WRITERS.get(set_type)
        if writer is _write58:
            _write58(fh, dset, mode, force_double=force_double)
        elif writer is not None:
            writer(fh, dset)
        # else: unsupported data-set - do nothing


if __name__ == '__main__':
    pass