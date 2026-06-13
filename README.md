# Proctoring_datasets
# Authors and copyright
Supervisor and main author V. A. Parhomenko, co-author I. A. Epishin. Сopyright © V. A. Parhomenko, I. A. Epishin.

# General Description
This repository contains datasets obtained after processing the raw data. To obtain datasets from the root of the datasets folder, use the scripts/convert_logs_to_datasets_ours script. It converts the raw data into training datasets.

The scripts/train_models script trains models using the obtained GazeDS, HeadDS, and GazeHeadDS datasets.

The scripts/convert_logs_to_datasets_others script converts the raw data into five datasets, compiled based on the descriptions provided by the authors of other papers. These datasets are located in the datasets/others folder.

The scripts/compare_on_our_dataset script trains models based on the datasets from the datasets/others folder. The algorithms used are also those recommended by the authors of the papers. 

The scripts/compare_on_others_datasets script trains models based on two publicly available datasets. No input is required; the datasets are loaded automatically.

# Warranty
The contributors provide no warranty for the use of this software. Use it at your own risk.

# License
This project is open for use in educational purposes and is licensed under the MIT License.
