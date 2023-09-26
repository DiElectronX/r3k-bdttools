# BDT (& more) Tools for Testing Run 3 R(K)

## Setting up Environment
You  can set up a computing environment with the necessary Python libraries using a Conda virtual environment `conda env create -f environment.yml`, or if you have a working python distribution, using pip `pip install --upgrade pip && pip install -r requirements.txt`.

## Running the scripts
Preprocessing assumes inout files are formatted according to [Run 3 CMGTools repo](https://github.com/DiElectronX/cmgtools-lite).

### Preparing Data Inputs
    python prepare_data_inputs.py -m split -j 10 -i <input dir> -o <output dir>
### Preparing MC Inputs
    python prepare_mc_inputs.py -m split -i <input dir> -o <output dir>

### Training Model
    python train_bdt.py --modelname <output dir and model name> --sigfile <signal file> --bkgfile <background file>
### Adding Scores to Measurement File

Manually choose options for how to make measurement:

    python measure_bdt.py  --model <model pickle file> --measurefile <measurement file>

Use params from directory log gile
    python measure_bdt.py  --fromdir <training output dir> --measurefile <measurement file>
