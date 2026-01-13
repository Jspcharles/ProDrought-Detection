# Data Directory Overview

This directory documents the structure, provenance, and role of all datasets used in this study.
Due to the large volume, raw data files are not hosted in this repository. Instead, this directory provides a transparent description of the data organisation and processing logic to support full reproducibility.

All datasets used in this study are derived from publicly available sources (SILO), and can be reconstructed using the preprocessing notebooks and scripts provided in the repository.

```data/
├── Aus_Shap_File/
│
├── Data_for_CD_Label/
│
├── Data_for_Modelling/
│   └── Annual_DB/
│       ├── 1_AUS/
│       │   ├── Max_Temp/
│       │   ├── Min_Temp/
│       │   ├── Radiation/
│       │   ├── Rainfall/
│       │   └── Vapour_Pressure_Deficit/
│       │
│       ├── 2_QLD_Cropped/
│       │   ├── Max_Temp/
│       │   ├── Min_Temp/
│       │   ├── Radiation/
│       │   └── Rainfall/
│       │
│       └── 3_QLD_Cleaned_Data/
│           ├── Max_Temp/
│           ├── Min_Temp/
│           ├── Radiation/
│           └── Rainfall/
│
├── Data_for_SPI/
│   ├── Monthly_Rainfall_for_SPI/
│   ├── Rainfall_2000/
│   │   ├── Monthly_Rainfall/
│   │   └── Region_Rainfall/
│   │
│   ├── Rainfall_Cropped_QLD/
│   │
│   ├── Region_Separated_Monthly_Rainfall/
│   │   ├── Brigalow_Belt/
│   │   ├── Cape_York_Peninsula/
│   │   ├── Central_Queensland_Coast/
│   │   ├── Channel_Country/
│   │   ├── Desert_Uplands/
│   │   ├── Einasleigh_Uplands/
│   │   ├── Gulf_Plains/
│   │   ├── Mitchell_Grass_Downs/
│   │   ├── Mulga_Lands/
│   │   ├── New_England_Tableland/
│   │   ├── Northwest_Highlands/
│   │   ├── Southeast_Queensland/
│   │   └── Wet_Tropics/
│   │
│   ├── Region_wise_SPI/
│   │   ├── All_Region_SPI_CSVs/
│   │   ├── Brigalow_Belt/
│   │   ├── Cape_York_Peninsula/
│   │   ├── Central_Queensland_Coast/
│   │   ├── Channel_Country/
│   │   ├── Desert_Uplands/
│   │   ├── Einasleigh_Uplands/
│   │   ├── Gulf_Plains/
│   │   ├── Mitchell_Grass_Downs/
│   │   ├── Mulga_Lands/
│   │   ├── New_England_Tableland/
│   │   ├── Northwest_Highlands/
│   │   ├── Southeast_Queensland/
│   │   └── Wet_Tropics/
│   │
│   └── Vapour_Pressure_Deficit/
│       └── Region_Separated_Monthly_VPD/
│           ├── Brigalow_Belt/
│           ├── Cape_York_Peninsula/
│           ├── Central_Queensland_Coast/
│           ├── Channel_Country/
│           ├── Desert_Uplands/
│           ├── Einasleigh_Uplands/
│           ├── Gulf_Plains/
│           ├── Mitchell_Grass_Downs/
│           ├── Mulga_Lands/
│           ├── New_England_Tableland/
│           ├── Northwest_Highlands/
│           ├── Southeast_Queensland/
│           └── Wet_Tropics/
```


## Folder Descriptions
`Aus_Shap_File/`

Contains vector boundary files used for spatial operations, including masking, cropping, and region-based aggregation. These shapefiles define the spatial extent of Australia, Queensland, and bioregional boundaries used throughout the analysis.

`Data_for_CD_Label/`

Stores intermediate tabular outputs used during consecutive drought (CD) labelling, including threshold definition, calibration, and sensitivity analysis. These files support the rule-based identification of drought onset, termination, and recovery periods.

`Data_for_Modelling/`

Contains climate variables prepared for use as deep-learning model inputs.

`1_AUS/`

National-scale annual climate datasets downloaded from source providers prior to spatial subsetting.

`2_QLD_Cropped/`

Climate datasets spatially clipped to Queensland boundaries using the provided shapefiles.

`3_QLD_Cleaned_Data/`

Quality-controlled and harmonised Queensland datasets used directly in model training and evaluation.

Each climate variable is organised into a dedicated subfolder:

`Max_Temp` – maximum temperature

`Min_Temp` – minimum temperature

`Radiation` – surface solar radiation

`Rainfall` – precipitation

Vapour_Pressure_Deficit – atmospheric moisture demand

`Data_for_SPI/`

Contains rainfall inputs and derived drought indices used for Standardised Precipitation Index (SPI) analysis.

`Monthly_Rainfall_for_SPI/`

Monthly rainfall data prepared as input for SPI calculation.

`Rainfall_2000/`

Baseline rainfall data used to initialise SPI time series.

`Rainfall_Cropped_QLD/`

Queensland-specific rainfall datasets derived from national products.

`Region_Separated_Monthly_Rainfall/`

Monthly rainfall data aggregated by bioregion to support region-wise SPI computation.

`Region_wise_SPI/`

SPI outputs generated separately for each bioregion, including summary CSV files and intermediate SPI products.

`Vapour_Pressure_Deficit/Region_Separated_Monthly_VPD/`

Monthly VPD datasets aggregated by bioregion, used to characterise vegetation stress and support recovery analysis.