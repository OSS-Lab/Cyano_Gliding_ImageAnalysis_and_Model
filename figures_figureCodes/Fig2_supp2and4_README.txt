Workflow for Fig2--supplements 2 and 4:

Raw data is in data/glass_movement_video/, which contains the first 500 frames of a 1000 frame video as an example. 
These figure supplements show results from two videos, each showing a different slide prepared from the same culture on the same day. Full videos are available on request. Vid1 has 1000 frames and vid2 has 500 frames.

We run Fig2_supp2and4_1segment.m to segment the image, then Fig2_supp2and4_2track.py to track the identified particles. The output of 2track.py is a data frame containing the filament trajectories. The two data frames (for each video) are provided in data/glass_movement_results/ 
(we have not provided the intermediate .mat and .csv files that are produced and saved by the Matlab segmentation script.)

Fig2_supp2and4_3analyse_plot.py performs the analysis and plots Fig2--supplements 2 and 4. 