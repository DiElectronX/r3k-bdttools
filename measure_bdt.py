import os
import argparse
import numpy as np
import uproot as ur
from xgboost import DMatrix
from utils import load_dir_args, check_rm_files, edit_filename, load_bdt


def evaluate_bdt(bdt, args, bdt_cols, output_dict, selection, score_skim=None):
    isMC = ('MC' in args.measurefile) or args.mc
    out_cols = output_dict['common'] + (output_dict['mc'] if isMC else output_dict['data'])

    modelname = edit_filename(
        args.filepath, prefix='measurement', suffix=args.label)
    check_rm_files([modelname, modelname.replace('.pkl', '.root')])

    with ur.open(args.measurefile) as datafile:
        data_dict = datafile['mytree'].arrays(
            bdt_cols, cut=selection if selection else None, library='np')
        out_branches = datafile['mytree'].arrays(
            out_cols, cut=selection if selection else None, library='np')
        data = np.transpose(np.stack(list(data_dict.values())))

    if selection:
        print('Additional Preselection Cuts:')
        for k, val in selection.items():
            print(f'  -{k}: {val}')

    if ('.pkl' in args.format) or ('.pickle' in args.format):
        decisions = np.array([x[1] for x in bdt.predict_proba(data)], dtype=np.float64)
    else:
        decisions = np.array(bdt.predict(DMatrix(data)), dtype=np.float64)

    out_branches['xgb'] = decisions

    if score_skim:
        mask = decisions > score_skim
        for arr in out_branches.values():
            arr = arr[mask]

    with ur.recreate((modelname.split('.')[0] if '.' in modelname else modelname)+'.root') as outfile:
        outfile['mytreefit'] = out_branches


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
    parser.add_argument('--log', dest='log', default=None,
                        type=str, help='logfile to read model parameters from')
    parser.add_argument('--mc', dest='mc', action='store_true',
                        help='flag for specifying MC sample (will already look for "MC" in filename)')
    parser.add_argument('--format', dest='format', default='.json',
                        help='format of saved model file')
    args, unknown = parser.parse_known_args()

    # default input variables
    features = ['Bprob', 'BsLxy', 'L2iso/L2pt', 'Bcos', 'Kiso/Kpt', 'LKdz', 'LKdr', 'Passymetry', 'Kip3d/Kip3dErr', 'L1id', 'L2id']

    # preseelction cuts (should already be applied)
    preselection = ''
    score_skim = None

    # output branch files
    output_dict = {
        'common' : ['Bmass', 'Mll', 'Bprob', 'BsLxy', 'Kpt', 'L1pt', 'L2pt', 'Keta', 'L1eta', 'L2eta',
                    'Bcos', 'LKdz', 'LKdr', 'L1id', 'L2id', 'Kiso', 'L1iso', 'L2iso', 'KLmassD0',
                    'Passymetry', 'Kip3d', 'Kip3dErr',
                   ],
        'data'   : [],
        'mc'     : ['trig_wgt'],
    }

    # load from directory
    if args.fromdir:
        load_dir_args(args)
        features = args.features

    # load model
    bdt = load_bdt(args)

    assert os.path.exists(args.measurefile)

    # evaluate model
    evaluate_bdt(bdt, args, features, output_dict, preselection, score_skim)
