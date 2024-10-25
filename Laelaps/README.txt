The Folder contains intermediate results for postprocessing and all the scripts to reproduce the results of Laelaps: An Energy-Efficient Seizure Detection Algorithm from Long-term Human iEEG Recordings without False Alarms.
Folders:
	- intermediate_results: contains intermediate results into two types.
	- Total_period_randomiM: prediction of HD kernel without postprocessing.
	- Pred_rel_mean: smoothing version after AveragePooling, kernel = 10x1, stride = 1.
- scripts:
	- bias_constant_compute: computation of the correction bias constant for tr computation.
	- HD_model: functions for iEEG_HD_analysis_prediction
	- iEEG_HD_analysis_prediction: prediction with the HD functions
	- postprocess: apply both tr and tc to smoothed recordings.
	- smoothing: smooth the recordings with average pooling
	- tr_computation: compute the tr threshold
	- Load_files_seizures: shows how to load the correct file with the seizure starting from the info file of each patient.
	- dataLoader: contains information and function to load data from patient
- scripts/output:
	- bias_output_orig: output of bias_constant_compute;
	- trcomp_output_orig: output of tr_computation;
	- postproc_output_orig: output of postprocess;