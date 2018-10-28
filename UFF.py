# -*- coding: utf-8 -*-
"""
Created on Fri Apr 06 17:27:08 2018

@author: kb-
Write time-series to UFF file
"""
import numpy as np,pyuff

def write(file,mask,fs,data,point,ordinate_axis_units_lab,abscissa_axis_lab='Time',abscissa_axis_units_lab='s',binary=0):
    
    d=mask
    d['abscissa_axis_lab'] = abscissa_axis_lab
    d['abscissa_axis_units_lab'] = abscissa_axis_units_lab
#    d['abscissa_force_unit_exp']
    d['abscissa_inc'] = 1/fs
#    d['abscissa_len_unit_exp']
#    d['abscissa_min']
#    d['abscissa_spacing']
#    d['abscissa_spec_data_type']
#    d['abscissa_temp_unit_exp']
    d['binary'] = binary
    d['data'] = data
#    d['func_id']
#    d['func_type']
#    d['id1']
    d['id2'] = 'Pt='+point+';'
#    d['id3']
#    d['id4']
#    d['id5']
#    d['load_case_id']
    d['num_pts'] = len(data)
#    d['ord_data_type']
#    d['orddenom_axis_lab']
#    d['orddenom_axis_units_lab']
#    d['orddenom_force_unit_exp']
#    d['orddenom_len_unit_exp']
#    d['orddenom_spec_data_type']
#    d['orddenom_temp_unit_exp']
    d['ordinate_axis_lab'] = point
    d['ordinate_axis_units_lab'] = ordinate_axis_units_lab
#    d['ordinate_force_unit_exp']
#    d['ordinate_len_unit_exp']
#    d['ordinate_spec_data_type']
#    d['ordinate_temp_unit_exp']
#    d['ref_dir']
#    d['ref_ent_name']
#    d['ref_node']
#    d['rsp_dir']
    d['rsp_ent_name'] = point
#    d['rsp_node']
#    d['type']
#    d['ver_num']
    d['x'] = np.linspace(0, len(data)/fs, len(data), endpoint=False)
#    d['z_axis_axis_lab']
#    d['z_axis_axis_units_lab']
#    d['z_axis_force_unit_exp']
#    d['z_axis_len_unit_exp']
#    d['z_axis_spec_data_type']
#    d['z_axis_temp_unit_exp']
#    d['z_axis_value']
    
    uffwrite = pyuff.UFF(file)
    uffwrite._write_set(d,'add')
