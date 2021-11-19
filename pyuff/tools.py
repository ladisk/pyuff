import pyuff
import numpy as np

def _opt_fields(dict, fields_dict):
    """Sets the optional fields of the dict dictionary. 
    
    Optionally fields are given in fields_dict dictionary.
    
    :param dict: dictionary to be updated
    :param fields_dict: dictionary with default values
    """

    for key in fields_dict:
        #             if not dict.has_key(key):
        if not key in dict:
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
    fields = {}
    n_fields_req = len(names)
    fields_from_line = []
    fields_out = {}

    # Extend the line if shorter than 80 chars
    ll = len(line)
    if ll < 80:
        line = line + ' ' * (80 - ll)
    # Parse the line for fields
    si = 0
    for n in range(0, len(widths)):
        fields_from_line.append(line[si:si + widths[n]].strip())
        si += widths[n]
    # Check for the number of fields,...
    n_fields = len(fields_from_line)
    if (n_fields_req < n_fields) or (min_values > n_fields):
        raise Exception('Error parsing header section; too many or to less' + \
                            'fields found')
    # Mandatory fields
    for key, n in zip(names[:min_values], range(0, min_values)):
        if types[n] == -1:
            pass
        elif types[n] == 1:
            fields_out.update({key: fields_from_line[n]})
        elif types[n] == 2:
            fields_out.update({key: int(fields_from_line[n])})
        else:
            fields_out.update({key: float(fields_from_line[n])})
    # Optional fields
    for key, n in zip(names[min_values:n_fields], range(min_values, n_fields)):
        try:
            if types[n] == -1:
                pass
            elif types[n] == 1:
                fields_out.update({key: fields_from_line[n]})
            elif types[n] == 2:
                fields_out.update({key: int(fields_from_line[n])})
            else:
                fields_out.update({key: float(fields_from_line[n])})
        except ValueError:
            if types[n] == 1:
                fields_out.update({key: ''})
            elif types[n] == 2:
                fields_out.update({key: 0})
            else:
                fields_out.update({key: 0.0})
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

def check_dict_for_none(dataset):
    dataset1 = {}
    for k, v in dataset.items():
        if v is not None:
            dataset1[k] = v

    return dataset1


def convert_dataset_2412_to_82(datasets):
    """Converts the dataset 2412 to dataset 82"""

    datasets_82 = []
    nodes = list()
    for i in range(0, len(datasets)):
        if (datasets[i]["type"] == 2412) and ("quad" in datasets[i].keys()):
            for j in range(len(datasets[i]["quad"]["nodes_nums"])):
                nodes.append(datasets[i]["quad"]["nodes_nums"][j][0])
                nodes.append(datasets[i]["quad"]["nodes_nums"][j][1])
                nodes.append(0)
                nodes.append(datasets[i]["quad"]["nodes_nums"][j][1])
                nodes.append(datasets[i]["quad"]["nodes_nums"][j][2])
                nodes.append(0)
                nodes.append(datasets[i]["quad"]["nodes_nums"][j][2])
                nodes.append(datasets[i]["quad"]["nodes_nums"][j][3])
                nodes.append(0)
                nodes.append(datasets[i]["quad"]["nodes_nums"][j][3])
                nodes.append(datasets[i]["quad"]["nodes_nums"][j][0])
                nodes.append(0)
            nodes = np.array(nodes)
            dataset_82 = pyuff.prepare_82(
                trace_num=i,
                n_nodes=len(nodes),
                nodes=nodes,
                color=0,
                id="",
                return_full_dict=True,
            )
            datasets_82.append(dataset_82)

    return datasets_82
