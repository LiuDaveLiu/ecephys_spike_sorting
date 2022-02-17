import os
import shutil
import subprocess
import numpy as np

from helpers import SpikeGLX_utils
from helpers import log_from_json
from helpers import run_one_probe
from create_input_json import createInputJson


# script to run CatGT, KS2, postprocessing and TPrime on data collected using
# SpikeGLX. The construction of the paths assumes data was saved with
# "Folder per probe" selected (probes stored in separate folders) AND
# that CatGT is run with the -out_prb_fld option

# -------------------------------
# -------------------------------
# User input -- Edit this section
# -------------------------------
# -------------------------------

# brain region specific params
# can add a new brain region by adding the key and value for each param
# can add new parameters -- any that are taken by create_input_json --
# by adding a new dictionary with entries for each region and setting the 
# according to the new dictionary in the loop to that created json files.


refPerMS_dict = {'default': 1.5, 'cortex': 1.5, 'medulla': 1.5, 'thalamus': 1.5}

# threhold values appropriate for KS2, KS2.5
ksTh_dict = {'default':'[10,4]', 'cortex':'[10,4]', 'medulla':'[10,4]', 'thalamus':'[10,4]'}
# threshold values appropriate for KS3.0
# ksTh_dict = {'default':'[9,9]', 'cortex':'[9,9]', 'medulla':'[9,9]', 'thalamus':'[9,9]'}



# -----------
# Input data
# -----------
# Name for log file for this pipeline run. Log file will be saved in the
# output destination directory catGT_dest
# If this file exists, new run data is appended to it
logName = 'practice2_log.csv'

# Raw data directory = npx_directory
# run_specs = name, gate, trigger and probes to process
npx_directory = r'I:\ephys\DL034'

# Each run_spec is a list of 4 strings:
#   undecorated run name (no g/t specifier, the run field in CatGT)
#   gate index, as a string (e.g. '0')
#   triggers to process/concatenate, as a string e.g. '0,400', '0,0 for a single file
#           can replace first limit with 'start', last with 'end'; 'start,end'
#           will concatenate all trials in the probe folder
#   probes to process, as a string, e.g. '0', '0,3', '0:3'
#   brain regions, list of strings, one per probe, to set region specific params
#           these strings must match a key in the param dictionaries above.

run_specs = [									
                        #['20210308', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20210309', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20210310', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20210311', '0', '0,0', '0:1', ['default', 'default'] ]
                        #['20210322', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20210323', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20210324', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20210325', '0', '0,0', '0', ['default'] ]
                        #['20210329', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20210330', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20210331', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20210409', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20210410', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20210411', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20210412', '0', '0,0', '0:1', ['default', 'default'] ]
                        #['20210422', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20210423', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20210424', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20210425', '0', '0,0', '0:1', ['default', 'default'] ]
                        #['20210409', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20210410', '0', '0,0', '0:1', ['default','default'] ],
                        #['20210411', '0', '0,0', '0:1', ['default', 'default'] ]
                        #['20210412', '0', '0,0', '0:1', ['default', 'default'] ]
                        #['20210411', '0', '0,0', '0:1', ['default', 'default'] ]
                        #['20210417', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20210418', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20210419', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20210420', '0', '0,0', '0', ['default'] ]
                        #['20210325', '0', '0,0', '0', ['default'] ]
                        #['20210430', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20210501', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20210502', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20210503', '0', '0,0', '0:1', ['default', 'default'] ]
                        #['20210506', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20210507', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20210508', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20210509', '0', '0,0', '0:1', ['default', 'default'] ]
                        #['20210526', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20210523', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20210524', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20210525', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20210522', '0', '0,0', '0:1', ['default', 'default'] ]
                        #['20210513', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20210514', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20210515', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20210516', '0', '0,0', '0:1', ['default', 'default'] ]
                        #['20210531', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20210530', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20210529', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20210601', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20210527', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20210528', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20210529', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20210530', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20210531', '0', '0,0', '0:1', ['default', 'default'] ]
                        #['20210521', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20210522', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20210523', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20210524', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20210525', '0', '0,0', '0', ['default'] ]
                        #['20210604', '0', '0,0', '0', ['default'] ],
                        #['20210605', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20210606', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20210607', '0', '0,0', '0:1', ['default', 'default'] ]
                        #['20210608', '0', '0,0', '0:1', ['default', 'default'] ]
                        #['20210609', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20210610', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20210611', '0', '0,0', '0:1', ['default', 'default'] ]
                        #['20210614', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20210615', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20210616', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20210617', '0', '0,0', '0:1', ['default', 'default'] ]
                        #['20210618', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20210619', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20210620', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20210621', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20210622', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20210623', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20210624', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20210625', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20210626', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20210627', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20210628', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20210629', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20210630', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20210701', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20210702', '0', '0,0', '0:1', ['default', 'default'] ]
                        #['20210707', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20210708', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20210709', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20210710', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20210711', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20210712', '0', '0,0', '1', ['default'] ]
                        #['20210713', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20210714', '0', '0,0', '0:1', ['default', 'default'] ]
                        #['20211217', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20211218', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20211219', '0', '0,0', '0:1', ['default', 'default'] ]
                        #['20211220', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20211221', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20211222', '0', '0,0', '0:1', ['default', 'default'] ]
                        #['20211223', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20211224', '0', '0,0', '0', ['default'] ]
                        #['20211222', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20211223', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20211224', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20211225', '0', '0,0', '0:1', ['default', 'default'] ]
                        #['20211229', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20211230', '0', '0,0', '0', ['default'] ],
                        #['20211231', '0', '0,0', '0:1', ['default', 'default'] ]
                        #['20220104', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20220105', '0', '0,0', '0', ['default'] ],
                        #['20220106', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20220107', '0', '0,0', '0', ['default'] ]
                        #['20220109', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20220110', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20220111', '0', '0,0', '0:1', ['default', 'default'] ]
                        #['20220112', '0', '0,0', '0:1', ['default', 'default'] ]
                        #['20220113', '0', '0,0', '0:1', ['default', 'default'] ]
                        #['20220106', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20220107', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20220109', '0', '0,0', '0:1', ['default', 'default'] ]
                        #['20220110', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20220111', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20220112', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20220113', '0', '0,0', '0:1', ['default', 'default'] ]
                        #['20220120', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20220121', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20220122', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20220123', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20220124', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20220125', '0', '0,0', '0:1', ['default', 'default'] ]
                        #['20220126', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20220127', '0', '0,0', '0:1', ['default', 'default'] ]
                        #['20220203', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20220204', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20220205', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20220206', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20220207', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20220208', '0', '0,0', '0:1', ['default', 'default'] ],
                        #['20220209', '0', '0,0', '0:1', ['default', 'default'] ]
                        ['20220204', '0', '0,0', '0:1', ['default', 'default'] ],
                        ['20220205', '0', '0,0', '0:1', ['default', 'default'] ],
                        ['20220206', '0', '0,0', '0:1', ['default', 'default'] ],
                        ['20220207', '0', '0,0', '0:1', ['default', 'default'] ],
                        ['20220208', '0', '0,0', '0:1', ['default', 'default'] ],
                        ['20220209', '0', '0,0', '0:1', ['default', 'default'] ],
                        ['20220210', '0', '0,0', '0:1', ['default', 'default'] ],
                        ['20220211', '0', '0,0', '0', ['default'] ]
                        #['20181125', '0', '0,0', '0', ['default'] ]
                        #['20181125', '0', 'start,end', '0', ['default'] ]
                        #['20210818', '0', '0,0', '0', ['default'] ]
]

# ------------------
# Output destination
# ------------------
# Set to an existing directory; all output will be written here.
# Output will be in the standard SpikeGLX directory structure:
# run_folder/probe_folder/*.bin
catGT_dest = r'I:\kilosort_datatemp\DL034'

# ------------
# CatGT params
# ------------

run_CatGT = True   # set to False to sort/process previously processed data.
#run_CatGT = False   # set to False to sort/process previously processed data.


# CAR mode for CatGT. Must be equal to 'None', 'gbldmx', or 'loccar'
car_mode = 'loccar'
#car_mode = 'None'
# inner and outer radii, in um for local comman average reference, if used
loccar_min = 40
loccar_max = 160

# CatGT commands for bandpass filtering, artifact correction, and zero filling
# Note 1: directory naming in this script requires -prb_fld and -out_prb_fld
# Note 2: this command line includes specification of edge extraction
# see CatGT readme for details
# these parameters will be used for all runs
catGT_cmd_string = '-prb_fld -out_prb_fld -aphipass=300 -tshift -gfix=0.40,0.10,0.02'
#catGT_cmd_string = '-prb_fld -out_prb_fld -apfilter=butter,16,250,0 -tshift -gfix=0.40,0.10,0.02'
#catGT_cmd_string = '-prb_fld -out_prb_fld -gblcar -apfilter=butter,16,250,0 -gfix=0.40,0.10,0.02'
#catGT_cmd_string = '-prb_fld -out_prb_fld -gblcar -gfix=0.40,0.10,0.02'
#catGT_cmd_string = '-prb_3A -no_run_fld -t_miss_ok -gblcar -apfilter=butter,16,250,0 -gfix=0.40,0.10,0.02'

ni_present = True
#ni_present = False
#ni_extract_string = '-XD=8,0,500 -XA=0,1,0.1,0'
ni_extract_string = '-XD=-1,0,500 -XA=0,1,0.1,0 -XA=1,1,0.1,2'



# ----------------------
# KS2 or KS25 parameters
# ----------------------
# parameters that will be constant for all recordings
# Template ekmplate radius and whitening, which are specified in um, will be 
# translated into sites using the probe geometry.
ks_remDup = 0
ks_saveRez = 1
ks_copy_fproc = 0
ks_templateRadius_um = 163
ks_whiteningRadius_um = 163
ks_minfr_goodchannels = 0.1


# ----------------------
# C_Waves snr radius, um
# ----------------------
c_Waves_snr_um = 160

# ----------------------
# psth_events parameters
# ----------------------
# extract param string for psth events -- copy the CatGT params used to extract
# events that should be exported with the phy output for PSTH plots
# If not using, remove psth_events from the list of modules
event_ex_param_str = 'XA=0,1,0.1,0'

# -----------------
# TPrime parameters
# -----------------
runTPrime = True   # set to False if not using TPrime
#runTPrime = False   # set to False if not using TPrime
sync_period = 1.0   # true for SYNC wave generated by imec basestation
toStream_sync_params = 'SY=0,-1,6,500'  # copy from the CatGT command line, no spaces
#niStream_sync_params = 'XD=-1,0,500'   # copy from the CatGT comman line, set to None if no Aux data, no spaces
niStream_sync_params = 'XD=4,0,500'   # copy from the CatGT comman line, set to None if no Aux data, no spaces

# ---------------
# Modules List
# ---------------
# List of modules to run per probe; CatGT and TPrime are called once for each run.
modules = [
			'kilosort_helper',
            'kilosort_postprocessing',
            'noise_templates',
            'psth_events',
            'mean_waveforms',
            'quality_metrics'
			]

json_directory = r'C:\Kilosort-2.5\temp'

# -----------------------
# -----------------------
# End of user input
# -----------------------
# -----------------------

# delete the existing CatGT.log
try:
    os.remove('CatGT.log')
except OSError:
    pass

# delete existing Tprime.log
try:
    os.remove('Tprime.log')
except OSError:
    pass

# delete existing C_waves.log
try:
    os.remove('C_Waves.log')
except OSError:
    pass

# check for existence of log file, create if not there
logFullPath = os.path.join(catGT_dest, logName)
if not os.path.isfile(logFullPath):
    # create the log file, write header
    log_from_json.writeHeader(logFullPath)
    
    


for spec in run_specs:

    session_id = spec[0]

    
    # Make list of probes from the probe string
    prb_list = SpikeGLX_utils.ParseProbeStr(spec[3])
    
    # build path to the first probe folder; look into that folder
    # to determine the range of trials if the user specified t limits as
    # start and end
    run_folder_name = spec[0] + '_g' + spec[1]
    prb0_fld_name = run_folder_name + '_imec' + prb_list[0]
    prb0_fld = os.path.join(npx_directory, run_folder_name, prb0_fld_name)
    first_trig, last_trig = SpikeGLX_utils.ParseTrigStr(spec[2], prb_list[0], spec[1], prb0_fld)
    trigger_str = repr(first_trig) + ',' + repr(last_trig)
    
    # loop over all probes to build json files of input parameters
    # initalize lists for input and output json files
    catGT_input_json = []
    catGT_output_json = []
    module_input_json = []
    module_output_json = []
    session_id = []
    data_directory = []
    
    # first loop over probes creates json files containing parameters for
    # both preprocessing (CatGt) and sorting + postprocessing
    
    for i, prb in enumerate(prb_list):
            
        #create CatGT command for this probe
        print('Creating json file for CatGT on probe: ' + prb)
        # Run CatGT
        catGT_input_json.append(os.path.join(json_directory, spec[0] + prb + '_CatGT' + '-input.json'))
        catGT_output_json.append(os.path.join(json_directory, spec[0] + prb + '_CatGT' + '-output.json'))
        
        # build extract string for SYNC channel for this probe
        sync_extract = '-SY=' + prb +',-1,6,500'
        
        # if this is the first probe proceessed, process the ni stream with it
        if i == 0 and ni_present:
            catGT_stream_string = '-ap -ni'
            extract_string = sync_extract + ' ' + ni_extract_string
        else:
            catGT_stream_string = '-ap'
            extract_string = sync_extract
        
        # build name of first trial to be concatenated/processed;
        # allows reaidng of the metadata
        run_str = spec[0] + '_g' + spec[1] 
        run_folder = run_str
        prb_folder = run_str + '_imec' + prb
        input_data_directory = os.path.join(npx_directory, run_folder, prb_folder)
        fileName = run_str + '_t' + repr(first_trig) + '.imec' + prb + '.ap.bin'
        continuous_file = os.path.join(input_data_directory, fileName)
        metaName = run_str + '_t' + repr(first_trig) + '.imec' + prb + '.ap.meta'
        input_meta_fullpath = os.path.join(input_data_directory, metaName)
        
        print(input_meta_fullpath)
         
        info = createInputJson(catGT_input_json[i], npx_directory=npx_directory, 
                                       continuous_file = continuous_file,
                                       kilosort_output_directory=catGT_dest,
                                       spikeGLX_data = True,
                                       input_meta_path = input_meta_fullpath,
                                       catGT_run_name = spec[0],
                                       gate_string = spec[1],
                                       trigger_string = trigger_str,
                                       probe_string = prb,
                                       catGT_stream_string = catGT_stream_string,
                                       catGT_car_mode = car_mode,
                                       catGT_loccar_min_um = loccar_min,
                                       catGT_loccar_max_um = loccar_max,
                                       catGT_cmd_string = catGT_cmd_string + ' ' + extract_string,
                                       extracted_data_directory = catGT_dest
                                       )      
        
        #create json files for the other modules
        session_id.append(spec[0] + '_imec' + prb)
        
        module_input_json.append(os.path.join(json_directory, session_id[i] + '-input.json'))
        
        
        # location of the binary created by CatGT, using -out_prb_fld
        run_str = spec[0] + '_g' + spec[1]
        run_folder = 'catgt_' + run_str
        prb_folder = run_str + '_imec' + prb
        data_directory.append(os.path.join(catGT_dest, run_folder, prb_folder))
        fileName = run_str + '_tcat.imec' + prb + '.ap.bin'
        continuous_file = os.path.join(data_directory[i], fileName)
 
        outputName = 'imec' + prb + '_ks2'

        # kilosort_postprocessing and noise_templates moduules alter the files
        # that are input to phy. If using these modules, keep a copy of the
        # original phy output
        if ('kilosort_postprocessing' in modules) or('noise_templates' in modules):
            ks_make_copy = True
        else:
            ks_make_copy = False

        kilosort_output_dir = os.path.join(data_directory[i], outputName)

        print(data_directory[i])
        print(continuous_file)
        
        # get region specific parameters
        ks_Th = ksTh_dict.get(spec[4][i])
        refPerMS = refPerMS_dict.get(spec[4][i])
        print( 'ks_Th: ' + repr(ks_Th) + ' ,refPerMS: ' + repr(refPerMS))

        info = createInputJson(module_input_json[i], npx_directory=npx_directory, 
	                                   continuous_file = continuous_file,
                                       spikeGLX_data = True,
                                       input_meta_path = input_meta_fullpath,
									   kilosort_output_directory=kilosort_output_dir,
                                       ks_make_copy = ks_make_copy,
                                       noise_template_use_rf = False,
                                       catGT_run_name = session_id[i],
                                       gate_string = spec[1],
                                       probe_string = spec[3],  
                                       ks_remDup = ks_remDup,                   
                                       ks_finalSplits = 1,
                                       ks_labelGood = 1,
                                       ks_saveRez = ks_saveRez,
                                       ks_copy_fproc = ks_copy_fproc,
                                       ks_minfr_goodchannels = ks_minfr_goodchannels,                  
                                       ks_whiteningRadius_um = ks_whiteningRadius_um,
                                       ks_Th = ks_Th,
                                       ks_CSBseed = 1,
                                       ks_LTseed = 1,
                                       ks_templateRadius_um = ks_templateRadius_um,
                                       extracted_data_directory = catGT_dest,
                                       event_ex_param_str = event_ex_param_str,
                                       c_Waves_snr_um = c_Waves_snr_um,                               
                                       qm_isi_thresh = refPerMS/1000
                                       )   

        # copy json file to data directory as record of the input parameters 
       
        
    # loop over probes for processing.    
    for i, prb in enumerate(prb_list):  
        
        run_one_probe.runOne( session_id[i],
                 json_directory,
                 data_directory[i],
                 run_CatGT,
                 catGT_input_json[i],
                 catGT_output_json[i],
                 modules,
                 module_input_json[i],
                 logFullPath )
                 
        
    if runTPrime:
        # after loop over probes, run TPrime to create files of 
        # event times -- edges detected in auxialliary files and spike times 
        # from each probe -- all aligned to a reference stream.
    
        # create json files for calling TPrime
        session_id = spec[0] + '_TPrime'
        input_json = os.path.join(json_directory, session_id + '-input.json')
        output_json = os.path.join(json_directory, session_id + '-output.json')
        
        # build list of sync extractions to send to TPrime
        im_ex_list = ''
        for i, prb in enumerate(prb_list):
            sync_extract = '-SY=' + prb +',-1,6,500'
            im_ex_list = im_ex_list + ' ' + sync_extract
            
        print('im_ex_list: ' + im_ex_list)     
        
        info = createInputJson(input_json, npx_directory=npx_directory, 
    	                                   continuous_file = continuous_file,
                                           spikeGLX_data = True,
                                           input_meta_path = input_meta_fullpath,
                                           catGT_run_name = spec[0],
    									   kilosort_output_directory=kilosort_output_dir,
                                           extracted_data_directory = catGT_dest,
                                           tPrime_im_ex_list = im_ex_list,
                                           tPrime_ni_ex_list = ni_extract_string,
                                           event_ex_param_str = event_ex_param_str,
                                           sync_period = 1.0,
                                           toStream_sync_params = toStream_sync_params,
                                           niStream_sync_params = niStream_sync_params,
                                           tPrime_3A = False,
                                           toStream_path_3A = ' ',
                                           fromStream_list_3A = list()
                                           ) 
        
        command = "python -W ignore -m ecephys_spike_sorting.modules." + 'tPrime_helper' + " --input_json " + input_json \
    		          + " --output_json " + output_json
        subprocess.check_call(command.split(' '))  
    

