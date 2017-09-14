PATH_SOURCE =  '../test_output/gic/ttf_timelapse_20160621_S01_001.ds'
DELAY=20
Z_SLICE=16

colorize_tifs :
	python colorize_tifs.py $(PATH_SOURCE) --timelapse

blend_channels :
	python blend_channels.py $(PATH_SOURCE)/rgb --timelapse

timelapse_signal_gif :
	convert -delay $(DELAY) $(PATH_SOURCE)/gray_adjusted/"*_signal*.tif"[$(Z_SLICE)] $(PATH_SOURCE)/timelapse_signal.gif

timelapse_dna_gif :
	convert -delay $(DELAY) $(PATH_SOURCE)/rgb/"*_dna_*.tif"[$(Z_SLICE)] $(PATH_SOURCE)/timeplapse_dna.gif

timelapse_combo_gif :
	convert -delay $(DELAY) $(PATH_SOURCE)/rgb/"*_combo*.tif"[$(Z_SLICE)] $(PATH_SOURCE)/timeplapse_combo.gif

