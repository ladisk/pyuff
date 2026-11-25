import pyuff
import numpy as np
import warnings

def _opt_fields(dict, fields_dict):
    """Sets the optional fields of the dict dictionary.

    Optionally fields are given in fields_dict dictionary. A field that is
    missing or None (e.g. from a prepare function with return_full_dict=True)
    gets its default value.

    :param dict: dictionary to be updated
    :param fields_dict: dictionary with default values
    """

    for key in fields_dict:
        if dict.get(key) is None:
            dict.update({key: fields_dict[key]})
    return dict

def _parse_header_line(line, min_values, widths, types, names):
    """Parses the given line (a record in terms of UFF file) and returns all
    the fields. 

    Fields are split according to their widths as given in widths.
    min_values specifies how many values (fields) are mandatory to read
    from the line. Also, number of values found must not exceed the
    number of fields requested by fieldsIn.
    On the output, a dictionary of field names and corresponding
    filed values is returned.
    
    :param line: a string representing the whole line.
    :param min_values: specifies how many values (fields) are mandatory to read
        from the line
    :param widths: fields widths to be read
    :param types: field types 1=string, 2=int, 3=float, -1=ignore the field
    :param names: a list of key (field) names
    """
    n_fields = len(widths)
    # Check for the number of fields,...
    if (len(names) < n_fields) or (min_values > n_fields):
        raise Exception('Error parsing header section; too many or to less' +                             'fields found')
    # Extend the line if shorter than 80 chars
    if len(line) < 80:
        line = line.ljust(80)
    fields_out = {}
    si = 0
    for n in range(n_fields):
        width = widths[n]
        field_type = types[n]
        if field_type == -1:
            si += width
            continue
        field = line[si:si + width].strip()
        si += width
        if n < min_values:
            # Mandatory field
            if field_type == 1:
                fields_out[names[n]] = field
            elif field_type == 2:
                fields_out[names[n]] = int(field)
            else:
                fields_out[names[n]] = float(field)
        else:
            # Optional field
            try:
                if field_type == 1:
                    fields_out[names[n]] = field
                elif field_type == 2:
                    fields_out[names[n]] = int(field)
                else:
                    fields_out[names[n]] = float(field)
            except ValueError:
                if field_type == 2:
                    fields_out[names[n]] = 0
                else:
                    fields_out[names[n]] = 0.0
    return fields_out

def _write_record(fh, values, formats, multiline=False, fstring=True):
    """Write a record to the open file using the specified formats.

    :param fh: opened file
    :param values: str or list, values to be written to the fields of the record
    :param formats: str or list, formats for each record
    :param multiline: if True, newline character is inserted after every value
    :param fstring: if True, f-string formatting is used (currently, only fstring=True
        is supported)    
    """

    if type(values) not in [list, tuple]:
        values = [values]
    if type(formats) not in [list, tuple]:
        formats = [formats]
    
    
    to_write = ''

    for v, f in zip(values, formats):
        if fstring:
            if type(v) == str:
                align = '<'
            else:
                align = '>'
            to_write = to_write + f'{v:{align}{f}}'

            if multiline:
                to_write = to_write + '\n'
    
    if not multiline:
        to_write = to_write + '\n'

    fh.write(to_write)

def _fit_string(value, width, field, set_type):
    """Returns the value as a string of at most width characters.

    A longer string does not fit the fixed-width field of the record; it is
    truncated with a warning naming the data-set and the field.

    :param value: value to be written to a text field
    :param width: width of the field (characters)
    :param field: name of the field (key of the dataset dictionary)
    :param set_type: data-set number
    """
    value = str(value)
    if len(value) > width:
        warnings.warn('Data-set %s: %s is longer than %i characters and is truncated to %r.'
                      % (set_type, field, width, value[:width]), stacklevel=5)
        value = value[:width]
    return value

def _columns_as_lists(n, *columns):
    """Returns the columns as lists of (Python) values, for fast formatting of rows.

    Raises IndexError if a column has less than n values (the rows are written up
    to n; longer columns are cut to n values).
    """
    out = []
    for column in columns:
        values = column.tolist() if hasattr(column, 'tolist') else list(column)
        if not isinstance(values, list) or len(values) < n:
            raise IndexError('a column of the data-set has less values than nodes')
        out.append(values[:n])
    return out


def _column_arrays(n, *columns):
    """Returns the first n values of each column as an array (for pyuff._vector_text).

    Raises IndexError if a column has less than n values (as indexing them did).
    """
    out = []
    for column in columns:
        values = np.asarray(column)
        if values.ndim != 1 or len(values) < n:
            raise IndexError('a column of the data-set has less values than nodes')
        out.append(values[:n])
    return out


def check_dict_for_none(dataset):
    dataset1 = {}
    for k, v in dataset.items():
        if v is not None:
            dataset1[k] = v

    return dataset1


def convert_dataset_2412_to_82(datasets):
    """Converts the dataset 2412 to dataset 82"""

    datasets_82 = []
    for i in range(0, len(datasets)):
        if (datasets[i]["type"] == 2412) and ("quad" in datasets[i].keys()):
            # the four edges of each quad, each a move (two nodes) followed by 0 (pen up);
            # the nodes of each data-set 2412 make their own trace line
            quads = np.asarray([q[:4] for q in datasets[i]["quad"]["nodes_nums"]], dtype=int).reshape(-1, 4)
            edges = np.zeros((len(quads), 4, 3), dtype=int)
            edges[:, :, 0] = quads
            edges[:, :, 1] = np.roll(quads, -1, axis=1)
            nodes = edges.ravel()
            dataset_82 = pyuff.prepare_82(
                trace_num=i,
                n_nodes=len(nodes),
                nodes=nodes,
                color=0,
                id="NONE",  # the identification line must not be blank
                return_full_dict=True,
            )
            datasets_82.append(dataset_82)

    return datasets_82
