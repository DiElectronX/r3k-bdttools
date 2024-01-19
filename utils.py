import os
import logging
import ast
import numpy as np
import multiprocessing as mp
from xgboost import Booster
from joblib import dump,load
from glob import glob
from pathlib import Path


class Logger():
    def __init__(self, filepath, verbose=True):        
        self.filepath = filepath
        self.verbose = verbose
        if os.path.exists(self.filepath):
            os.remove(self.filepath)
        
        log_format = '%(levelname)s | %(asctime)s | %(message)s'
        logging.basicConfig(level=logging.INFO, format=log_format)

    def log(self, string):
        with open(self.filepath, 'a+') as f:
            f.write(string+'\n')
        if self.verbose:
            logging.info(string)


def make_file_name(args):
    name_blocks = [
        args.modelname,
        'Nsig'+str(args.stop_sig)+'_Nbkg' +
        str(args.stop_bkg) if (args.stop_sig or args.stop_bkg) else '',
        args.label,
    ]
    return '_'.join(filter(None, name_blocks))


def save_model(model, args, formats, logger):
    output_name = make_file_name(args)

    if '.pkl' in formats:
        name = os.path.join(args.outdir, output_name+'.pkl')
        dump(model, name)
        if logger:
            logger.log(f'Saving Model {name}')
    if '.text' in formats:
        name = os.path.join(args.outdir, output_name+'.text')
        booster = model.get_booster()
        booster.dump_model(name, dump_format='text')
        if logger:
            logger.log(f'Saving Model {name}')
    if '.json' in formats:
        name = os.path.join(args.outdir, output_name+'.json')
        model.save_model(name)
        if logger:
            logger.log(f'Saving Model {name}')
    if '.txt' in formats:
        name = os.path.join(args.outdir, output_name+'.txt')
        model.save_model(name)
        if logger:
            logger.log(f'Saving Model {name}')

def load_bdt(args):
    args.filepath = args.model if args.format in args.model else args.model+args.format
    assert os.path.exists(args.filepath)

    if ('pkl' in args.format) or ('pickle' in args.format):
        return load(args.filepath)
    else:
        bdt = Booster()
        bdt.load_model(args.filepath)
        return bdt 

def preprocess_files(input_files, nparts, total):
    filelist = [input_files] if input_files.endswith('.root') else glob(input_files+'/**/*.root',recursive=True)[:total]
    if nparts==1:
        outfiles = filelist
    else:
        outfiles = np.array_split(np.array(filelist), nparts if nparts!=-1 else mp.cpu_count())

    if not outfiles: 
        raise ValueError('Invalid input path/file')
    return outfiles


def load_dir_args(args):
    if args.log:
        logname = args.log
    else:
        logs = [f for f in os.listdir(args.fromdir) if (('log_' in f) and ('.txt' in f))]
        if len(logs)==1:
            logname = logs[0]
        else:
            raise KeyError('Multiple viable log files, use "--log" flag to pick one')

    with open(os.path.join(args.fromdir, logname)) as f:
        for line in f:
            if 'Decay: ' in line:
                args.decay = line.split('Decay: ', 1)[1].strip()
            if 'Inputs: ' in line:
                args.features = ast.literal_eval(
                    line.split('Inputs: ', 1)[1].strip())
            if ('Saving Model ' in line) and (args.format in line):
                args.model = line.split('Saving Model ', 1)[1].strip()

    print(f'Parsing {args.fromdir} Directory')
    print(f'Measuring {args.decay} Decay')
    print(f'Using Model {args.model}')
    print(f'Using Input Vector {args.features}')


def check_rm_files(files=[]):
    for fl in files:
        if os.path.isfile(fl):
            os.system('rm '+fl)


def edit_filename(path, prefix='', suffix=''):
    path = Path(path)
    return os.path.join(str(path.parent), (prefix+'_' if prefix else '') + str(path.stem) + ('_'+suffix if suffix else '') + str(path.suffix))

