from evaluation_functions import *
import os.path

model_dirs = [ '/mnt/External1/Yael/comparison/fusion_clean',
	   '/mnt/External1/Yael/comparison/fusion_stn_first',
	   '/mnt/External1/Yael/comparison/fusion_stn_last']

out_path = '/mnt/External1/Yael/comparison/'
datasets = ['full', 'common', 'challenging', 'test']

for test_data in datasets:
	model_errors = []
	model_names = []

	for i,model_dir in enumerate(model_dirs):
		model_name = model_dir.split('/')[-1]
		err_path = os.path.join(model_dir, 'logs/nme_statistics', model_name+'_'+
				test_data+'_nme.npy')
		err = np.load(err_path)
		model_names.append(model_name)
		model_errors.append(err)
	print_ced_compare_methods(
			method_errors=tuple(model_errors),
			method_names=tuple(model_names),
			test_data=test_data,
			log_path=out_path,
			save_log=True)

