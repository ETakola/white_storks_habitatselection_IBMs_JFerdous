+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Study: An individual-based model for white storks (Ciconia ciconia) in Germany during breeding season



Contact: Jannatul Ferdous (Jannatul.ferdous@tu-dresden.de)



What: Description of scripts used in the study

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

**GEE\_scripts** is a text file containing the links and codes from the Google Earth Engine to extract the LULC and NDVI-NDWI data as points and raster layers

**Folder "Scripts\_for\_IBM\_STRIDE"** contains following scripts. They are Python codes written in spyder 6.0

**STRIDE: Stork Trajectory and Resource-based Individual Dynamics model**

**File name: STRIDE v2.3.py**

* The main model code for the simulated movement
* Input files: (Can be found in folder Input\_files)

&#x09;- MPIAB\_inside\_data.xlsx

&#x09;- IBM\_bird\_random\_effects\_ndvi\_model.csv

&#x09;- IBM\_fixed\_effects\_ndvi\_model

&#x09;- Raster Files (Please refer to [GitHub repository](https://github.com/ETakola/white_storks_habitatselection_IBMs_JFerdous/tree/main/Spatial%2520Resources) for the files)

&#x09;	- LC\_2012.tif

&#x09;	- NDVI\_2012\_JulAug.tif

&#x09;	- MNDWI\_2012\_JulAug.tif

&#x09;	- SHEI\_2012.tif

&#x09;	- ED\_2012.tif

**###These raster files are only for creating the base environment; all the files in the GitHub folder will be required as a model input###**

* Output files:3 folders for each year

  * Steps: Contains separate files for each bird\_id recording their 30 minute interval steps along with the environmental parameters
  * Figures: Figures for checking if all the years are running properly, bird movement trajectories and the number of birds based on their final status at the end of the simulation year
  * Summaries: Summary of all the birds from the year of simulation. It contains the final status of the birds, their net energy, and if any of the birds has shifted.



For computational and time efficiency, the validation part is done in two separate codes.

**Filename: Sensitivity\_analysis\_better\_site\_score.py**

* Contains the code for the sensitivity analysis using the variable "BETTER\_SITE\_SCORE"
* As it calls the model directly from the STRIDE v2.2, please check the model name and path before execution
* Input file: STRIDE v2.3.py
* Output files: Same files as the main model with the selected repetitions.



**Filename: Sensitivity\_analysis\_extended.py**

* Contains the code for sensitivity analysis using the two variables: "MIN\_SHIFT\_DISTANCE\_M" and "BREED\_MIN\_EXCESS"
* As it calls the model directly from the STRIDE v2.3, please check the model name and path before execution
* Input file: STRIDE v2.3.py
* Output files: Same files as the main model with the selected repetitions.



**The other scripts are R-codes for rest of the analysis.**

**Filename: used\_available\_point\_creation.R**: code for creating 10 available points for each true GPS point from the filtered MPIAB data

* Input file: MPIAB\_inside\_data.csv
* Output file: used\_avaiable\_datapoints.rds / used\_avaiable\_datapoints.csv

**Filename: Landscapemetrics\_point\_based\_value\_extraction.py**: script for calculating landscape metrics using the used-available points

* Input files: final\_clean\_dataset\_updated.rds and the folder path for the raster files ==> **in\_dir  <- *"\[path]"***
* Output files: All\_variables\_RSF.rds / All\_variables\_RSF.csv

**###The columns in the All\_variables\_RSF.rds were scaled using the scale() function in R and later were combind using rbind to create All\_variables\_RSF\_scales.rds###**

**Filename: clogit\_mit\_validation.R:** script for conditional logistic regression RSF model with matched ID

* Input file: All\_variables\_RSF\_scaled.rds
* Output files: 

  * matching\_structure\_check.csv: Checks the matched used-available structure for each fix\_id. Reports the number of rows, used points, available points, and birds per choice set. Used to verify that each matched set contains one used point, at least one available point, and belongs to one bird.
  * RSF\_dataset\_summary\_for\_text.csv: Provides manuscript-ready dataset summary statistics based on valid matched choice sets. Includes number of valid choice sets, used points, available points, used ratio, and number of birds.
  * retained\_LULC\_classes.csv: Lists the LULC classes retained after removing sparse classes. Used to document the categorical habitat classes included in the RSF models.
  * RSF\_multicollinearity\_VIF\_ndwi.txt: Stores variance inflation factor values for the model containing ndwi, ED\_scaled, and LULC\_class. Used to assess multicollinearity for the NDWI-based candidate structure.
  * RSF\_multicollinearity\_VIF\_ndvi.txt: Stores variance inflation factor values for the model containing ndvi, ED\_scaled, and LULC\_class. Used to assess multicollinearity for the NDVI-based candidate structure.
  * RSF\_numeric\_predictor\_correlation\_matrix.csv: Provides the correlation matrix among numeric predictors: ndvi, ndwi, ED\_scaled, SHDI\_scaled, SHEI\_scaled, and PD\_scaled. Used to document predictor correlations and potential collinearity.
  * clogit\_model\_AIC\_comparison.csv: Compares all candidate population-level clogit models using AIC. Candidate models combine one vegetation predictor, ndvi or ndwi, with one landscape metric, ED\_scaled, SHDI\_scaled, SHEI\_scaled, or PD\_scaled, plus LULC\_class.
  * population\_best\_model\_fixed\_effects.csv: Stores coefficient estimates for the best-AIC population model. Includes term names, beta estimates, standard errors, z-values, p-values, and exponentiated coefficients.
  * IBM\_fixed\_effects\_best\_model.csv: Provides fixed-effect coefficients from the best-AIC population model. Contains model terms and coefficient values only.
  * per\_bird\_data\_counts\_best\_model.csv: Summarises data availability per bird for the best-AIC model. Includes number of rows, choice sets, used points, and available points per bird.
  * per\_bird\_model\_diagnostics\_best\_model.csv: Reports diagnostics for bird-specific clogit models fitted using the best-AIC model predictors. Includes model success/failure status, reason for failure if applicable, sample size, AIC, and log-likelihood.
  * per\_bird\_coefficients\_best\_model\_long.csv: Stores bird-specific coefficient estimates for the best-AIC model in long format, with one row per bird and predictor term. Includes beta, standard error, z-value, p-value, and exponentiated coefficient.
  * per\_bird\_coefficients\_best\_model.csv: Stores bird-specific coefficient estimates for the best-AIC model in wide format, with one row per bird and predictor coefficients as separate columns.
  * IBM\_bird\_random\_effects\_best\_model.csv: Provides bird-level coefficients from the best-AIC model. This file contains bird-specific coefficient estimates/slopes, not deviations from the population coefficient.
  * intraspecific\_variability\_summary\_best\_model.csv: Summarises intraspecific variation in bird-specific coefficients for the selected best-AIC model predictors. Includes mean, standard deviation, minimum, quartiles, maximum, and counts of positive and negative coefficients.
  * all\_population\_model\_summaries.txt: Stores the full summary output for all fitted candidate population clogit models. Used as a complete model archive for checking estimates, standard errors, and model diagnostics.
  * **IBM\_fixed\_effects\_ndvi\_model.csv:** Provides fixed-effect coefficients from the manually selected ecological model: used \~ ndvi + ED\_scaled + LULC\_class + strata(fix\_id) + cluster(bird\_id). This model is used later for validation and IBM simulations.
  * IBM\_bird\_specific\_slopes\_ndvi.csv: Stores bird-specific slopes for the NDVI + ED\_scaled model. Includes one row per bird with bird-specific ndvi and ED\_scaled coefficients.
  * intraspecific\_variability\_summary\_ndvi.csv: Provides a simple summary of bird-specific coefficient variation for the NDVI + ED\_scaled model. Includes mean and standard deviation for ndvi and ED\_scaled coefficients.
  * **IBM\_bird\_random\_effects\_ndvi\_model.csv:** Stores bird-specific deviations from the population-level NDVI + ED\_scaled coefficients. Includes re\_ndvi and re\_ED\_scaled, calculated as bird-specific coefficient minus population coefficient. This is the actual random effects for the NDVI model that was used in the IBM.
  * LOBO\_validation\_per\_bird.csv: Contains leave-one-bird-out cross-validation results for each test bird. Reports number of tested choice sets, top-1 accuracy, top-3 accuracy, and mean rank of the observed used point.
  * LOBO\_validation\_summary.csv: Summarises leave-one-bird-out validation across birds. Includes mean, standard deviation, minimum, and maximum top-1 accuracy, mean and standard deviation top-3 accuracy, and mean/sd rank.
  * ranking\_validation.csv: Reports in-sample ranking validation for the NDVI + ED\_scaled model. Evaluates whether the observed used point ranks first or within the top three alternatives within each matched choice set.
  * Boot\_validation\_summary.csv: Summarises bootstrap stability of the NDVI + ED\_scaled model coefficients. Includes bootstrap mean and 95% quantile interval for ndvi and ED\_scaled coefficients.
  * blocked\_kfold\_by\_bird\_fold\_summary.csv: Provides fold-wise results from bird-blocked k-fold cross-validation. Includes number of test birds, number of test choice sets, top-1 accuracy, top-3 accuracy, mean rank, and median rank per fold.
  * blocked\_kfold\_by\_bird\_overall\_summary.csv: Summarises bird-blocked k-fold cross-validation across folds. Includes mean, standard deviation, minimum, and maximum top-1 and top-3 accuracy, plus mean and standard deviation of mean rank.
  * bird\_specific\_variation\_summary\_for\_text.csv: Provides summary statistics for bird-specific ndvi and ED\_scaled coefficients. Includes minimum, quartiles, median, maximum, and standard deviation.
  * bird\_direction\_agreement\_with\_population.csv: Quantifies how many birds have coefficient directions matching or contrasting with the population-level NDVI + ED\_scaled effects. Reports counts and percentages for ndvi and ED\_scaled.
  * heterogeneity\_classification\_for\_text.csv: Classifies the degree of intraspecific heterogeneity based on sign reversal percentages and coefficient ranges for ndvi and ED\_scaled. Categories are weak, moderate, or substantial.
  * RSF\_validation\_summary\_for\_text.csv: Provides a compact validation summary combining leave-one-bird-out cross-validation, bird-blocked 5-fold cross-validation, and in-sample ranking. Reports mean rank, top-1 percentage, and top-3 percentage.
  * RSF\_bootstrap\_summary\_for\_text.csv: Provides bootstrap coefficient stability interpretation for ndvi and ED\_scaled. Classifies each coefficient as stable positive, stable negative, or variable/uncertain based on the bootstrap confidence interval.

**Filename: Landscapemetrics\_250m\_GeoTiffs.R**: script for creating GeoTiff rasters for landscapemetrics using the LULC rasters

**Filename: Figure\_2.R**: Script for Figure 2 in the manuscript

* Input file: All\_variables\_RSF\_scaled.rds
* Output file: Figure 2 from the main text in png, svg and pdf format at 600dpi resolution

**Filename: Figure\_3.R**: Script for Figure 3 in the manuscript

* Input file: merged\_simulated\_bird\_steps.rds
* Output file: Figure 3 from the main text in png, svg and pdf format at 600dpi resolution

**Filename: Figure\_4.R**: Script for Figure 4 in the manuscript

* Input file: merged\_simulated\_bird\_steps.rds
* Output file: Figure 4 from the main text in png, svg and pdf format at 600dpi resolution

**Filename: Figure\_5.R**: Script for Figure 5 in the manuscript

* Input file: merged\_simulated\_bird\_steps.rds
* Output file: Figure 5 from the main text in png, svg and pdf format at 600dpi resolution

**Filename: Figure\_6.R**: Script for Figure 6 in the manuscript

* Input file: merged\_summary\_all\_years.rds
* Output file: Figure 6 from the main text in png, svg and pdf format at 600dpi resolution

**Filename: Figure\_S1.R**: Script for Figure S1 in the supplementary section of the manuscript

* Input file: LOBO\_validation\_summary.csv, blocked\_kfold\_by\_bird\_overall\_summary.csv
* Output file: Figure S1 from the supplementary in png, svg and pdf format at 600dpi resolution

**Filename: Figure\_S2.R**: Script for Figure S2 in the supplementary section of the manuscript

* Input file: merged\_simulated\_bird\_steps.rds
* Output file: Figure S2 from the supplementary in png, svg and pdf format at 600dpi resolution

**Filename: Figure\_S3\_S4\_S5.R**: Script for Figure S3, S4, S5 in the supplementary section of the manuscript

* Input file:MIN\_SHIFT\_DISTANCE\_M\_year\_seed\_summary.rds, MIN\_BETTER\_SITE\_SCORE\_year\_seed\_summary.rds, BREED\_MIN\_EXCESS\_year\_seed\_summary.rds
* Output file: Figure S3, S4, and S5 from the supplementary in png, svg and pdf format at 600dpi resolution

❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋

**For the input files of figure 2, 3, 4, 5 and 6: please refer to the "input\_data" folder and for S1, S2, S3, S4 and S5 please find the input data in the "validation\_outputs" folder**

❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋❋

