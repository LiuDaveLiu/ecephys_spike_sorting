import os
import shutil

from create_input_json import createInputJson

#import ecephys_spike_sorting.modules.extract_from_npx as extract_from_npx

npx_files = [r'E:\704166722_380486_20180531_probeC\recording1.npx', 
             r'E:\704514354_380485_20180601_probeC\recording2.npx',
             r'E:\704514354_380485_20180601_probeB\recording1.npx']

json_directory = r'C:\Users\svc_neuropix\Documents\GitHub\ecephys_spike_sorting\ecephys_spike_sorting\scripts'

def copy_data_to_backup_drive(info):
	extracted_data_location = info['directories']['extracted_data_directory']
	new_location = os.path.join(r'E:', os.path.basename(extracted_data_location))
	shutil.move(extracted_data_location, new_location)

for npx_file in npx_files:

	probe_directory = os.path.dirname(npx_file)
	session_id = os.path.basename(probe_directory)

	input_json = os.path.join(json_directory, session_id + '-input.json')
	output_json = os.path.join(json_directory, session_id + '-output.json')

	info = createInputJson(npx_file, input_json)

	commands = ('mean_waveforms', 'quality_metrics')

	for command in commands:

		command = "python -m ecephys_spike_sorting.modules." + command + " --input_json " + input_json \
		          + " --output_json " + output_json

		os.system(command)

	#copy_data_to_backup_drive(info)




