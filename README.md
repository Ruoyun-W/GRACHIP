# GRACHIP: A Cell-type-specific 3D genome architecture predictor with DNA sequences and genomic features

This repository includes the code for GRACHIP, the Graph-based Chromatin Interaction Prediction model. With this model, you can obtain accurate chromatin architecture predictions with DNA sequences and genomic features. For additional information, refer to [our paper]().

## Installation

### Install GRACHIP dependency

``` bash
pip install pandas numpy argparse joblib matplotlib pyBigWig pyfaidx cooler
```
Our versions: 

pandas==2.0.2

numpy==1.24.3

argparse==1.1

joblib==1.2.0

matplotlib==3.7.1

pyBigWig==0.3.18

pyfaidx==0.8.1.1

cooler==0.10.2


#### Quick installation

Install via Anaconda

    conda install torch
    conda install pyg -c pyg

Install via pip

    pip install torch torch_geometric

our versions:

torch==2.0.0+cu117

torch_geometric==2.3.1


#### Install PyTorch based on your environment

You can install [PyTorch](https://pytorch.org/), [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) and/or CUDA based on your environment. We used torch 1.13, in our model development. [CUDA](https://developer.nvidia.com/cuda-12-1-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local) is recommended if supported by your GPU (in most cases, an NVIDIA GPU).

### Install DNABERT

DNABERT requires specific dependencies and setup. Please follow the installation instructions available at the [DNABERT](https://github.com/jerryji1993/DNABERT?tab=readme-ov-file) GitHub repository.

## Model weight and dataset download

Before running predictions, you need to download the pre-trained model weights and the test files.

### Download everything

[GRACHIP_data](https://drive.google.com/file/d/1Tvw17uN78lYvYoxcn9jI7jYQc7tsy1Qs/view?usp=drive_link)

### Download files separately

Model Weights:\
[6 featrue GRACHIP model](https://drive.google.com/file/d/1arfUbXBbubKxNxBseJRQHTm7MDqv59Pc/view?usp=drive_link)\
[ATAC+CTCF GRACHIP model](https://drive.google.com/file/d/1PjCMITIc_wkAgBlJ-s--IC-LVTku30LU/view?usp=drive_link)\
[ATAC+H3K4me GRACHIP model](https://drive.google.com/file/d/1KoKNocbhwcwUDJQSrYbS_XcO2F13PyaH/view?usp=drive_link)\
[ATAC+CTCF+H3K4me GRACHIP model](https://drive.google.com/file/d/1mGYFSjpV1r0_fUHuzeQPLZxSIAc4EeF0/view?usp=drive_link)

[DNABERT model weight](https://drive.google.com/file/d/1BJjqb5Dl2lNMg2warsFQ0-Xvn1xxfFXC/view?usp=sharing)\
[Edge input](https://drive.google.com/file/d/1QJIcx7bzNpT5kCleLD05Rsgq0R4O2CAE/view?usp=drive_link) This input is for human genome assembly hg38. If your predictions involve a different genome assembly, you should substitute the provided file with a .cool file at 10K resolution. For additional details, refer to the [cooler documentation](https://cooler.readthedocs.io/en/latest/index.html).\
[Encoder](https://drive.google.com/file/d/1F9u87x0UfwjmGrxG-VK_0uQ3Mo3cIIs6/view?usp=drive_link)\
[Test files for HFF](https://drive.google.com/file/d/1XtHakLFFjGC8a9WNhUXvYDPCB9E8HmaC/view?usp=drive_link)\
[hg38 for test](https://drive.google.com/file/d/1-Gc0RAmpp0zGppF9r24XEBvF1gR9WIni/view?usp=drive_link) You may use your own genome .fasta file (index it first).

### Use your own genomic data

We recommend using fold change over control bigwig files as input. For more information, please check the ENCODE pipeline for [ChIP-seq](https://www.encodeproject.org/chip-seq/transcription_factor/) and [ATAC-seq](https://www.encodeproject.org/atac-seq/). 

In a text file, specify the path of your bigwig files, each line for a file. for example:

    data/HFF/ATAC.bw
    data/HFF/CTCF.bw
    data/HFF/K4me3.bw
    data/HFF/K4me.bw
    data/HFF/K27ac.bw
    data/HFF/K27me3.bw

if you prefer to use only 2 or 3 features to make prediction, leave the rest lines with NAs. But make sure you load the right model for your feature combinations. for example:

    data/HFF/ATAC.bw
    data/HFF/CTCF.bw
    NA
    data/HFF/K4me.bw
    NA
    NA

### Other arguments

| Argument               | Description                            | Default Value        | Required |
|------------------|-------------------|------------------|------------------|
| `--genomic_data_paths` | Paths to genomic files                 | `./data/bwlist.txt`  | No       |
| `--output_path`        | Output directory                       | `./output`           | No       |
| `--genome_path`        | Reference genome .fa path              | `./data/hg38.fa`     | No       |
| `--model`              | Path to the model file                 | `./data/model.pt`    | No       |
| `--encoder`            | Path to the encoder file               | `./data/encoder.pt`  | No       |
| `--input_edge`         | Path to the input edge file            | `./data/input.cool`  | No       |
| `--start`              | Start site for genomic data extraction | None                 | Yes      |
| `--chrom`              | Target chromosome for analysis         | None                 | Yes      |
| `--DNABERT`            | Path to the DNABERT model directory    | `./data/6-new-12w-0` | No       |
| `--iterations`         | Number of iterations for prediction    | 10                   | No       |

## Prediction

Clone the repo
``` bash
git clone https://github.com/Ruoyun-W/GRACHIP
cd GRACHIP/ 
```

Run prediction.py file for predictions. A working example is as follows.

``` bash
python prediction.py \
    --chrom "chr2" \
    --start 196760000 \
    --genomic_data_paths "data/bwlist.txt" \
    --output "./output" \
    --genome_path "./data/hg38.fa" 
```

## Citation

If you use GRACHIP, please kindly cite our paper:

    @article {Wang2024.05.21.595047,
	author = {Wang, Ruoyun and Ma, Weicheng and Mohammadi, Aryan Soltani and Shahsavari, Saba and Vosoughi, Soroush and Wang, Xiaofeng},
	title = {Improving Cell-type-specific 3D Genome Architectures Prediction Leveraging Graph Neural Networks},
	elocation-id = {2024.05.21.595047},
	year = {2024},
	doi = {10.1101/2024.05.21.595047},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2024/05/21/2024.05.21.595047},
	eprint = {https://www.biorxiv.org/content/early/2024/05/21/2024.05.21.595047.full.pdf},
	journal = {bioRxiv}

