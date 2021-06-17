import numpy as np

from ..tools import UFFException, _opt_fields, _parse_header_line

def _write2414(fh, dset):
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


def _extract2414(blockData):
    # Extract analysis data - data-set 2414.
    dset = {'type': 2414}
    # Read data
    try:
        binary = False
        split_header = blockData.splitlines(True)[:15]  # Keep the line breaks!
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