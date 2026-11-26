import numpy as np
import sys, os
my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')

import pyuff

def test_read_write_read_given_data():
    test_read_write_read_given_data_base('./data/beam.uff')

def test_read_write_read_given_data_base(file=''):
    if file=='':
        return
    #read from file
    uff_read = pyuff.UFF(file)

    a = uff_read.read_datasets()
    if type(a)==list:
        types = np.array([_['type'] for _ in a])
        a = a[np.argwhere(types==151)[0][0]]

    #write to file
    save_to_file = './data/temp.uff'
    if os.path.exists(save_to_file):
        os.remove(save_to_file)
    _ = pyuff.UFF(save_to_file)
    _.write_datasets(a, 'add')

    #read back
    uff_read = pyuff.UFF(save_to_file)
    b = uff_read.read_datasets(0)

    if os.path.exists(save_to_file):
        os.remove(save_to_file)

    labels = [_ for _ in a.keys() if \
              any(_[-len(w):]==w for w in ['_lab', '_name', '_description',\
                                           '_created', '_saved', '_written',\
                                           'db_app', '_name'])]
    string_keys = ['program', 'description']
    exclude_keys = []

    string_keys = list(set(string_keys).union(set(labels)).difference(set(exclude_keys)))
    numeric_keys = list((set(a.keys()).difference(set(string_keys)).difference(set(exclude_keys))))


    for k in numeric_keys:
        print('Testing: ', k)
        np.testing.assert_array_almost_equal(a[k], b[k], decimal=3)
    for k in string_keys:
        print('Testing string: ', k, a[k])
        np.testing.assert_string_equal(a[k], b[k])


def test_write_read_151():
    save_to_file = './data/test.uff'

    dataset = pyuff.prepare_151(
        model_name='Model file name',
        description='Model file description',
        db_app='Program which created DB',
        date_db_created='27-Jan-16',
        time_db_created='14:38:15',
        version_db1=1,
        version_db2=2,
        file_type=0,
        date_db_saved='28-Jan-16',
        time_db_saved='14:38:16',
        program='OpenModal',
        date_db_written='29-Jan-16',
        time_db_written='14:38:17')
    
    dataset_out = dataset.copy()
    if save_to_file:
        if os.path.exists(save_to_file):
            os.remove(save_to_file)
        uffwrite = pyuff.UFF(save_to_file)
        uffwrite._write_dataset(dataset, 'add')
    
    a = dataset_out
    uff_read = pyuff.UFF(save_to_file)
    b = uff_read.read_datasets()
    if os.path.exists(save_to_file):
        os.remove(save_to_file)

    labels = [_ for _ in a.keys() if \
              any(_[-len(w):]==w for w in ['_lab', '_name', '_description',\
                                           '_created', '_saved', '_written',\
                                           'db_app', '_name'])]
    string_keys = ['program', 'description']
    exclude_keys = ['date_db_written', 'time_db_written']

    string_keys = list(set(string_keys).union(set(labels)).difference(set(exclude_keys)))
    numeric_keys = list((set(a.keys()).difference(set(string_keys)).difference(set(exclude_keys))))

    for k in numeric_keys:
        print('Testing: ', k)
        np.testing.assert_array_almost_equal(a[k], b[k])
    for k in string_keys:
        np.testing.assert_string_equal(a[k], b[k])

def test_prepare_151():
    dict_151 = pyuff.prepare_151(
        model_name='Model file name',
        description='Model file description',
        db_app='Program which created DB',
        date_db_created='27-Jan-16',
        time_db_created='14:38:15',
        version_db1=1,
        version_db2=2,
        file_type=0,
        date_db_saved='28-Jan-16',
        time_db_saved='14:38:16',
        program='OpenModal',
        date_db_written='29-Jan-16',
        time_db_written='14:38:17')
    x = sorted(list(dict_151.keys()))
    y = sorted(['type',
        'model_name',
        'description',
        'db_app',
        'date_db_created',
        'time_db_created',
        'version_db1',
        'version_db2',
        'file_type',
        'date_db_saved',
        'time_db_saved',
        'program',
        'date_db_written',
        'time_db_written'])
    np.testing.assert_array_equal(x,y)

    #empty dictionary test
    x2=pyuff.prepare_151()
    if 'type' not in x2.keys():
        raise Exception('Not correct keys')
    if x2['type'] != 151:
        raise Exception('Not correct type')

if __name__ == '__main__':
    test_write_read_151()

if __name__ == '__mains__':
    np.testing.run_module_suite()

def test_write_151_file_written(tmp_path):
    # the date and time the file was written are written as given, and read back as
    # date_file_written and time_file_written
    dset = pyuff.prepare_151(model_name='m', description='d', db_app='a', program='p',
                             date_db_written='29-Jan-16', time_db_written='14:38:17')
    fn = tmp_path / 'header.uff'
    pyuff.UFF(fn).write_datasets(dset, mode='overwrite')
    read = pyuff.UFF(fn).read_datasets(0)
    assert (read['date_file_written'], read['time_file_written']) == ('29-Jan-16', '14:38:17')

    # a data-set read from a file is written with its own values
    pyuff.UFF(fn).write_datasets(read, mode='overwrite')
    again = pyuff.UFF(fn).read_datasets(0)
    assert (again['date_file_written'], again['time_file_written']) == ('29-Jan-16', '14:38:17')
