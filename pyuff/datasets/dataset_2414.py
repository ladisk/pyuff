import numpy as np

from ..tools import _opt_fields, _parse_header_line, check_dict_for_none

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
        raise Exception('Error writing data-set #2414')


def _extract2414(block_data):
    """Extract analysis data - data-set 2414."""
    dset = {'type': 2414}
    # Read data
    try:
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
        if dset['analysis_type'] == 5:
            # frequency response 
            split_data = ''.join(block_data.splitlines(True)[15:])
            split_data = split_data.split()
            if dset['data_type'] == 5 and dset['number_of_data_values_for_the_data_component'] == 3:
                values = np.asarray([float(str) for str in split_data], 'd')
                dset['node_nums'] = np.array(values[::7].copy(), dtype=int)
                dset['x'] = values[1::7].copy()+values[2::7].copy()*1j
                dset['y'] = values[3::7].copy()+values[4::7].copy()*1j
                dset['z'] = values[5::7].copy()+values[6::7].copy()*1j   

        pass  
    except:
        raise Exception('Error reading data-set #2412')
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
    :param d: R15 F1, Data at this node (NDVAL real or complex values) **check**
    :param x: R15 F1,
    :param y: R15 F1,
    :param z: R15 F1,
    
    :param return_full_dict: If True full dict with all keys is returned, else only specified arguments are included
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
    if analysis_type not in (0, 1, 2, 3, 4, 6, 7, 9, None):
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
    if np.array(x).dtype != float and x != None:
        raise TypeError('x must be float')
    if np.array(y).dtype != float and y != None:
        raise TypeError('y must be float')
    if np.array(z).dtype != float and z != None:
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
        'real_part_of_modal_A_or_modal_mass,': real_part_of_modal_A_or_modal_mass, 
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

