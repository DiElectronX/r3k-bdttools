import os
import logging
import ast
import numpy as np
from joblib import dump
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
    with open(os.path.join(args.fromdir, f'{"log_"+args.label if args.label else "log"}.txt')) as f:
        for line in f:
            if 'Decay: ' in line:
                args.decay = line.split('Decay: ', 1)[1].strip()
            if 'Inputs: ' in line:
                args.features = ast.literal_eval(
                    line.split('Inputs: ', 1)[1].strip())
            if ('Saving Model ' in line) and ('.pkl' in line):
                args.model = line.split('Saving Model ', 1)[1].strip()

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

