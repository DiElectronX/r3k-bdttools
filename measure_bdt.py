import os
import argparse
import numpy as np
import uproot as ur
from pathlib import Path
from joblib import load
from utils import load_dir_args, check_rm_files, edit_filename


def evaluate_bdt(bdt, args, bdt_cols, output_dict, selection):
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

    decisions = np.array([x[1]
                         for x in bdt.predict_proba(data)], dtype=np.float64)
    out_branches['xgb'] = decisions

    with ur.recreate((modelname.split('.pkl')[0] if '.pkl' in modelname else modelname)+'.root') as outfile:
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
    parser.add_argument('--mc', dest='mc', action='store_true',
                        help='flag for specifying MC sample (will already look for "MC" in filename)')
    args, unknown = parser.parse_known_args()

    # select input variables
    # 'Slimmed' Inputs
    # features = ['Bprob','BsLxy','L2iso/L2pt','Kpt/Bmass','Bcos','Kiso/Kpt','LKdz','LKdr','Bpt/Bmass','Passymetry','Kip3d/Kip3dErr']
    # Otto / 'Run 2' Inputs
    # Missing: BBDphi, BTrkdxy2
    features = ['Bprob', 'BsLxy', 'L2iso/L2pt', 'Kpt/Bmass', 'Bcos', 'Kiso/Kpt', 'LKdz', 'LKdr', 'Bpt/Bmass', 'Passymetry', 'Kip3d/Kip3dErr', 'L1id', 'L2id']

    # Jay Inputs
    # Missing: ProbeTracks_DCASig[K], ProbeTracks_dzTrg[K], BToKEE_k_svip2d, BToKEE_l1_dxy_sig, BToKEE_l1_dzTrg, BToKEE_l2_dxy_sig, BToKEE_l2_dzTrg, BToKEE_llkDR
    # features = ['Biso/Bpt','L1L2dr','Bcos','Kpt/Bmass','L1pt/Bmass','L2pt/Bmass','Bpt/Bmass','Kiso/Kpt','Kip3d','L1iso/L1pt','L1id','L2iso/L2pt','L2id','BsLxy','Passymetry','Bprob']

    # preseelction cuts (should already be applied)
    preselection = ''

    # output branch files
    output_dict = {
    
        'common' : [
                'Bmass',
                'Mll',
                'Bprob',
                'BsLxy',
                'Npv',
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
        'mc' : [ 'trig_wgt' ],
    }
    # load from directory
    if args.fromdir:
        load_dir_args(args)
        features = args.features

    # load model
    args.filepath = args.model if '.pkl' in args.model else args.model+'.pkl'
    assert os.path.exists(args.filepath)
    assert os.path.exists(args.measurefile)
    bdt = load(args.filepath)

    # evaluate model
    evaluate_bdt(bdt, args, features, output_dict, preselection)
