
class UFFException(Exception):
    """An exception that prints a string describing the error.
    """

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return self.value


def _opt_fields(dict, fieldsDict):
    """Sets the optional fields of the dict dictionary. 
    
    Optionally fields are given in fieldsDict dictionary.
    
    :param dict: dictionary to be updated
    :param fieldsDict: dictionary with default values
    """

    for key in fieldsDict:
        #             if not dict.has_key(key):
        if not key in dict:
            dict.update({key: fieldsDict[key]})
    return dict

def _parse_header_line(line, minValues, widths, types, names):
    """Parses the given line (a record in terms of UFF file) and returns all
    the fields. 

    Fields are split according to their widths as given in widths.
    minValues specifies how many values (fields) are mandatory to read
    from the line. Also, number of values found must not exceed the
    number of fields requested by fieldsIn.
    On the output, a dictionary of field names and corresponding
    filed values is returned.
    
    :param line: a string representing the whole line.
    :param widths: fields widths to be read
    :param types: field types 1=string, 2=int, 3=float, -1=ignore the field
    :param names: a list of key (field) names
    """
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