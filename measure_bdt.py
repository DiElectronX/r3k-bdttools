import os
import argparse
import warnings
import numpy as np
import uproot as ur
from joblib import load
from utils import load_dir_args, check_rm_files, edit_filename

warnings.simplefilter(action='ignore', category=FutureWarning)

def evaluate_bdt(args):
    isMC = ('MC' in args.measurefile) or args.mc
    out_cols = args.output_dict['common'] + (args.output_dict['mc'] if isMC else args.output_dict['data'])
    presel = args.preselection if args.preselection else None
    modelname = edit_filename(args.filepath, prefix='measurement', suffix=args.label)
    check_rm_files([modelname, modelname.replace('.pkl', '.root')])

    with ur.open(args.measurefile) as datafile:
        data = datafile['mytree'].arrays(args.features, cut=presel, library='pd')
        output_branches = datafile['mytree'].arrays(out_cols, cut=presel, library='pd')

    if selection:
        print(f'Additional Preselection Cuts:\n{args.preselection}')

    output_branches['xgb'] = np.array(args.bdt.predict_proba(data)[:,1], dtype=np.float64)

    with ur.recreate((modelname.split('.pkl')[0] if '.pkl' in modelname else modelname)+'.root') as outfile:
        outfile['mytreefit'] = output_branches


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', dest='model',
                        default='model.pkl', type=str, help='model name or file')
    parser.add_argument('--measurefile', dest='measurefile', default=None,
                        type=str, help='file with events for measurement')
    parser.add_argument('--label', dest='label', default='',
                        type=str, help='label for root file')
    parser.add_argument('--decay', dest='decay', default='kee',
                        type=str, choices=['kmumu', 'kee'], help='decay type')
    parser.add_argument('--fromdir', dest='fromdir', default='',
                        type=str, help='load params from designated model directory')
    parser.add_argument('--mc', dest='mc', action='store_true',
                        help='flag for specifying MC sample (will already look for "MC" in filename)')
    args, unknown = parser.parse_known_args()

    # select input features
    args.features = ['Bprob', 'BsLxy', 'L2iso/L2pt', 'Kpt/Bmass', 'Bcos', 'Kiso/Kpt', 'LKdz', 'LKdr', 'Bpt/Bmass', 'Passymetry', 'Kip3d/Kip3dErr', 'L1id', 'L2id']

    # preseelction cuts (should already be applied)
    args.preselection = ''

    # output branch files
    args.output_dict = {
        'common' : [
                'Bmass',
                'Mll',
                'Bprob',
                'BsLxy',
                # 'Npv',
                'L1pt',
                'L2pt',
                'Kpt',
                'Bcos',
                'LKdz',
                'LKdr',
                'L2id',
                'Kiso',
                'L1id',
                'L1iso',
                'KLmassD0',
                'Passymetry',
                'Kip3d',
                'Kip3dErr',
                'L2iso',
                'Keta',
                'L2eta',
                'L1eta',
        ],
        'data' : [],
        'mc' : [],
    }
    # load from directory
    if args.fromdir:
        load_dir_args(args)

    # load model
    args.filepath = args.model if '.pkl' in args.model else args.model+'.pkl'
    assert os.path.exists(args.filepath)
    assert os.path.exists(args.measurefile)
    args.bdt = load(args.filepath)

    # evaluate model
    evaluate_bdt(args)
