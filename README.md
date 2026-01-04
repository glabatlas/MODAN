# MDNGM


MDNGM is a tool to identify cancer subtype-specific driver modules integrating multi-omics data.

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

**NOTE: Example data is included in this project, and this step is not required when using example data to varify if the program can execute properly. When using example data, please make sure that the preprocessed data in `./gene_sample_overlapped/` exist. You only need to perform this data prepare step when using your own data.**

Please put multi-omics data of cancer `CANCERNAME` into dictionary `./datasets/CANCERNAME/`. Gene mutation data, DNA methylation data, gene expression data and cancer subtype label are required.



- **Gene mutation data**
    - File name: `CANCERNAME_mut.csv`
    - Format: csv file, including row names and column names. Each row represents a gene and each column represents a cancer sample.

- **Gene expression data**
    - File name: `CANCERNAME_expr.csv`
    - Format: csv file, including row names and column names. Each row represents a gene and each column represents a cancer sample.

- **Gene methylation data**
    - File name: `CANCERNAME_methy.csv`
    - Format: csv file, including row names and column names. Each row represents a gene and each column represents a cancer sample.

- **Cancer sample subtype label**
    - File name: `CANCERNAME_subtype_label.csv`
    - Format: csv file with 2 columns and corresponding column names. The first column stores sample names, and the second column stores the corresponding subtype categories. Row names are not included.

After preparing data above, please modifify the `Dataset Config` part at the begining of `utils.py` file. In this file, `cancers` refers to cancer types (e.g., BRCA, GBM, LUAD), `subtypes` indicates subtypes for each cancer, and `mut_thres` indicates mutation frequency threshold, where genes with mutation rates below this threshold in the cancer dataset will be filtered out in this step.

For example, to identify subtype-specific driver modules in BRCA, GBM and LUAD, the configuration of this section can be set as follows:  

    cancers = ['BRCA', 'GBM', 'LUAD']

    subtypes = {'BRCA':['Basal','Her2','LumA','LumB'],
                'GBM':['Classical','Mesenchymal','Neural','Proneural'],
                'LUAD':['Bronchioid','Magnoid','Squamoid']}

    mut_thres = {'BRCA': 0.005,
                'GBM': 0.005,
                'GBM': 0.01,}

Then, please run the following command, and the script will prepare data of all cancers:
```
python ./data_prepare.py
```

# Module Detection

Please run the following command to integrate data and prepare for feature learning:
```
python ./diffusion.py
python ./exclusivity_network.py
```

Next, please perform feature learning for each subtype of each cancer type. For subtype `SUBTYPENAME` of cancer `CANCERNAME`, run this command for gene representation learning:
```
python ./feature_learning.py -c CANCERNAME -s SUBTYPENAME
```

After learning representation of all subtypes, please run this command to generate modules:
```
python ./clustering.py
```

The result for sbutype `SUBTYPENAME` of cancer `CANCERNAME` is stored in `./results/modules/CANCERNAME/SUBTYPENAME/final_results.csv`
# Reference Paper
Coming soon...