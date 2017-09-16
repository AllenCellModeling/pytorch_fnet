RUN_NAME = ttf_bf_dna_nohotscombo_0915
DATA = data/dataset_saves/ttf_bf_dna_nohotscombo.ds

long:
	python train_model.py --n_iter 500000 --buffer_size 30 --replace_interval -1 \
	--path_data $(DATA) \
	--batch_size 24 \
	--nn_module ttf_v8_nn \
	--run_name $(RUN_NAME)

snm :
	python train_model.py --n_iter 20 --buffer_size 1 --replace_interval -1 \
	--data_path data/nuc_mask \
	--data_set_module nucmaskdataset \
	--model_module snm_model \
	--nn_module snm_v0_nn \
	--lr 0.0001 \
	--gpu_ids 0 \
	--run_name snm_v0_tmp

gen_dataset_1 :
	python gen_dataset.py \
	--path_save data/dataset_saves/ttf_bf_dna_nohotscombo.ds \
	--name_target dna \
	--train_split 30 \
	--path_source data/no_hots_combo

	# --path_source data/dic_images
gen_dataset :
	python gen_dataset.py \
	--path_source data/timelapse_czis/20160621_S01_001.czi \
	--name_signal bf \
	--covfefe
	--train_split 0 \
	--path_save data/dataset_saves/ttf_timelapse_20160621_S01_001.ds \
	--name_target struct
gic :
	python integrated_cells.py \
	--path_source data/dataset_saves/ttf_timelapse_20160621_S01_001.ds \


#	--path_model saved_models/ttf_bf_dna_nohotscombo_30.p
# test_model :
# 	python test_model.py \
# 	--path_model saved_models/ttf_bf_dna_nohotscombo_30.p \
# 	--path_dataset data/dataset_saves/ttf_bf_dna_nohotscombo.ds \
# 	--n_images 2

test_model :
	python test_model.py \
	--path_source saved_models/ttf_bf_dna_no_relu \
	--n_images 4
