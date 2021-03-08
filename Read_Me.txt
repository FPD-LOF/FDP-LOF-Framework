This folder contains the code used for applying the FPD-LOF Framework. The FPD-LOF folder contains a Visual Studio project with all the functions used.
Running this project will create a results.pickle file that contains all info obtained from the raw data, most noteably a LOF_DF which contains the LOF scores for each intersection.
Of course, to run this a couple of configurations need to be edited to your needs - this can be done in the configs.json file.

The output of the FPD-LOF project for the Hoofddorp dataset are included as: "Hoofddorp_results.pickle".

To be able to use VLOG data, two notebooks for doing that are shared as well, inlcuded in the Process_VLOG folder.

To process the results of the AIMSUN simulation, the data and a notebook for processing them are included in hte Process_simulation_results folder.

Furthermore, Jupyter notebooks are included that were used to determine the correlations between the intersections, for the The Hague dataset and the Hoofddorp dataset. 
These files also include more analysis of the outliers.

Code regarding the Siemens traffic controller DIRECTOR cannot be shared.

Lastly, the code for producing the LSTM for the The Hague dataset is included in the "The_Hague_LSTM.ipynb" file.