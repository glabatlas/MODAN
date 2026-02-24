# MODAN

MODAN is a tool to identify cancer subtype-specific driver modules integrating multi-omics data.

## Dependencies
    python=3.8
    gensim
    stellargraph
    scikit-learn
    numpy
    pandas
    networkx
    scipy

## Data Prepare

**NOTE: This step is not required if you want to use demo data or reproduce the results in the paper.**

If you want to use your own data, please put Protein-Protein Interaction (PPI) network into `./datasets/CANCERNAME/ppi.csv`, then put gene mutation data, DNA methylation data, gene expression data of cancer `CANCERNAME` subtype `SUBTYPENAME` into dictionary `./datasets/CANCERNAME/SUBTYPENAME/`. Meanwhile, put DNA methylation data, gene expression data of normal samples into dictionary `./subtype_data/CANCERNAME/subtype/`. All datasets should contain the same genes.

- **PPI Network**
    - File name: `ppi_network.txt`
    - Format: txt file separated by `\t`, without row name or column name. Each line contains two gene symbols, representing the two nodes of an edge in the PPI network.

- **Gene mutation data**
    - File name: `mut.csv`(cancer samples)
    - Format: csv file, including row names and column names. Each row represents a gene and each column represents a cancer sample.

- **Gene expression data**
    - File name: `expr.csv`(cancer samples for specific subtype) or `expr_normal.csv`(normal samples)
    - Format: csv file, including row names and column names. Each row represents a gene and each column represents a cancer sample.

- **Gene methylation data**
    - File name: `methy.csv`(cancer samples for specific subtype) or `methy_normal.csv`(normal samples)
    - Format: csv file, including row names and column names. Each row represents a gene and each column represents a cancer sample.

# Module Detection

To identify driver modules for cancer `CANCER` with subtypes `Subtype1`, `Subtype2` and `Subtype3`, please run:

```
python ./main.py -c CANCER -s Subtype1 Subtype2 Subtype3
```

In this step, if you use demo data to check if this program runs correctly, please run:
```
python ./main.py -c BRCA_demo -s Basal Her2
```

To reproduce the results in the paper, extract the ZIP files for each cancer from `./datasets/` into subfolders within `./datasets/`, named after the respective cancer name.
 Then, for BRCA dataset, please run:
```
python ./main.py -c BRCA -s Basal Her2 LumA LumB
```
For GBM dataset, run:
```
python ./main.py -c GBM -s Classical Mesenchymal Neural Proneural
```
For LUAD dataset, run:
```
python ./main.py -c LUAD -s Bronchioid Magnoid Squamoid
```


# Results

The result for subtype `SUBTYPENAME` of cancer `CANCERNAME` is stored in `./results/modules/CANCERNAME/SUBTYPENAME/final_results.csv`

Additionally, trained models are saved in `./models/`, while the gene representations are saved in `./results/features`.

# Parameter Description
- `-c`, `--cancer`, (Required) Cancer type;
- `-s`, `--subtypes`, (Required) All subtypes of the cancer, including specific subtypes and subtypes used for comparison;
- `-p`, `--specific_subtypes`, (Optional) Subtypes used to identifying driver modules. If specified, only the driver modules of these specified subtypes will be identified.

# Code Structure
- **./datasets/** Datasets.
- **./datasets/BRCA_demo/** Demo datasets.
- **./datasets/\*.zip** Datasets used in paper.
- **./diffusion.py** Cancer-related gene network construction and network diffusion.
- **./feature_learning.py** Feature learning from constructed subtype-specific mutual exclusivity network.
- **./main.py** The main entry of this program.
- **./me_network.py** Subtype-specific mutual exclusivity network construction.
- **./module_detection.py** Module detection and optimization.
- **./utils.py** Metric calculation functions.

After running this program, the following folders will be generated:
- **./datasets_generate/** Predicted mutation effect states.
- **./me_network/** Mutual Exclusivity networks.
- **./models/** Trained node2vec models for gene representation learning.
- **./results/features/** Gene representations from node2vec, where each row represents a gene.
- **./results/modules/** Driver module results, where `initial_results.csv` represents initial driver module results after clustering, and `final_results.csv` represnets final driver results after module optimization.

# Reference Paper
Coming soon...

