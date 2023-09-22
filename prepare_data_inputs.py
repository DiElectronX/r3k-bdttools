import os
import sys
import argparse
import numpy as np
import uproot as ur
import awkward as ak
import multiprocessing as mp
from glob import glob

def preprocess_files(input_files, nparts, total):
    filelist = [input_files] if input_files.endswith('.root') else glob(input_files+'/**/*.root',recursive=True)[:total]
    if nparts==1:
        outfiles = filelist
    else:
        outfiles = np.array_split(np.array(filelist), nparts if nparts!=-1 else mp.cpu_count())

    if not outfiles: raise ValueError('Invalid input path/file')
    return outfiles

def preprocess_inputs(runFiles,ipart,args,branch_dict):
    # Branch Parameters
    MllcutMin = 1.05
    MllcutMax = 2.45
    Highq2Mllcut = 3.85
    MLeftSideMin = 4.8
    MLeftSideMax = 5.
    MRightSideMin = 5.4
    MRightSideMax = 5.6
    MBmin = 4.5
    MBmax = 6.

    if args.mode=='train':
        args.writeMeasurement = False
        args.useLowQ = True
        args.useBsideBands = True
    elif args.mode=='measure':
        args.writeMeasurement = True
        args.useLowQ = False
        args.useBsideBands = False

    if args.mode=='highq2train':
        args.writeMeasurement=False
        args.useHighQ=True
        args.useBsideBands=True

    tree_values={}
    ncands_branch = 'n'+branch_dict['candidate']
    needed_branches = list({ncands_branch} | branch_dict['cand_branches'].keys() | branch_dict['scalar_branches'].keys())
    for i, tree in enumerate(ur.iterate([runFile+':Events' for runFile in runFiles],needed_branches,cut=args.split,namedecode='utf-8',library='ak')):
        presel_mask = np.full(ak.flatten(tree[branch_dict['candidate']+'_fit_mass']).to_numpy().shape, True)
        # we want to rearrange leptons to make sure that they are properly sorted (pt or type). so the output leptonX will have two contributions from input leptonX and Y
        outl1_inl1_mask = np.copy(presel_mask)
        outl1_inl2_mask = np.copy(presel_mask)

        # Deal with scalars
        scalars = {}
        entries_per_evt = tree[ncands_branch]
        for br, outname in branch_dict['scalar_branches'].items():
            values_scl = tree[br]
            # array_scl = [scl for scl,itr in zip(values_scl, entries_per_evt) for i in range(itr)]
            # scalars[outname] = np.array(array_scl)
            scalars[br] = np.repeat(values_scl,entries_per_evt)

        sortby = 'leppt'
        if sortby=='eltype':
             id1 = ak.flatten(tree[branch_dict['candidate']+'_l1_isPF'])
             id2 = ak.flatten(tree[branch_dict['candidate']+'_l2_isPF'])
             outl1_inl1_mask = np.where(id1==1,1,0)
             outl1_inl2_mask = np.where(id2==1,1,0)
        else:
             pt1 = ak.flatten(tree[branch_dict['candidate']+'_fit_l1_pt'])
             pt2 = ak.flatten(tree[branch_dict['candidate']+'_fit_l2_pt'])
             outl1_inl1_mask = np.where(pt1>pt2,1,0) + np.where(pt1==pt2,1,0)
             outl1_inl2_mask = np.where(pt2>pt1,1,0)

        outl2_inl1_mask = 1 - outl1_inl1_mask
        outl2_inl2_mask = 1 - outl1_inl2_mask

        #remove infs from data
        copied_branches = {}
        inf_mask = np.full(len(outl1_inl1_mask),True)
        nan_mask = np.full(len(outl1_inl1_mask),True)
        for branch in branch_dict['cand_branches'].keys():
            copied_branches[branch] = ak.flatten(tree[branch])
            infs = np.argwhere(np.isinf(ak.flatten(tree[branch])))
            nans = np.argwhere(np.isnan(ak.flatten(tree[branch])))
            for idx in infs:
                inf_mask[idx] = False
                np.asarray(copied_branches[branch])[idx] = 0
            for idx in nans:
                nan_mask[idx] = False
                np.asarray(copied_branches[branch])[idx] = 0

        presel_mask = np.full(len(copied_branches[branch_dict['candidate']+'_fit_mass']),True)
        for key, cut in branch_dict['presel'].items():
            presel_mask = presel_mask * (copied_branches[key]>cut)

        presel_mask = presel_mask * inf_mask
        presel_mask = presel_mask * nan_mask

        mB_branch = copied_branches[branch_dict['candidate']+'_fit_mass']
        mll_branch = copied_branches[branch_dict['candidate']+'_mll_fullfit']
        if args.useBsideBands:
            presel_mask = presel_mask * ((mB_branch>MLeftSideMin) * (mB_branch<MLeftSideMax) + (mB_branch>MRightSideMin) * (mB_branch<MRightSideMax))
        else:
            presel_mask = presel_mask * ((mB_branch>MBmin) * (mB_branch<MBmax))
        if args.useLowQ:
            presel_mask = presel_mask * ((mll_branch>MllcutMin) * (mll_branch<MllcutMax))
        if args.useHighQ:
            presel_mask = presel_mask * (mll_branch>Highq2Mllcut)

        #eta cuts k, e1,e2
        k_eta_branch = copied_branches[branch_dict['candidate']+'_fit_k_eta']
        l1_eta_branch = copied_branches[branch_dict['candidate']+'_fit_l1_eta']
        l2_eta_branch = copied_branches[branch_dict['candidate']+'_fit_l2_eta']
        presel_mask = presel_mask * ((k_eta_branch<2.4) * (k_eta_branch>-2.4))
        presel_mask = presel_mask * ((l2_eta_branch<2.4) * (l2_eta_branch>-2.4) * (l1_eta_branch<2.4) * (l1_eta_branch>-2.4))

        outleps={}
        for br1, br2 in branch_dict['leppairs_branches'].items():
            #if exists take it from cleaned for e1
            if br1 in copied_branches.keys():
                outleps[br1] = copied_branches[br1] * outl1_inl1_mask + copied_branches[br2] * outl1_inl2_mask
            else:
                outleps[br1] = ak.flatten(tree[br1]) * outl1_inl1_mask + ak.flatten(tree[br2]) * outl1_inl2_mask
            #alternative id from tree -- pray not to have inf
            if 'Id' in br1 and sortby=='eltype':
                outleps[br2] = ak.flatten(tree[branchId_change[0]]) * outl2_inl1_mask + ak.flatten(tree[branchId_change[1]]) * outl2_inl2_mask
            else:
                #for e2
                if br2 in copied_branches.keys():
                    outleps[br2] = copied_branches[br1] * outl2_inl1_mask + copied_branches[br2] * outl2_inl2_mask
                else:
                    outleps[br2] = ak.flatten(tree[br1]) * outl2_inl1_mask + ak.flatten(tree[br2]) * outl2_inl2_mask

        for br, cut in branch_dict['leppairs_presel'].items():
            presel_mask = presel_mask * (outleps[br] > cut)

        output_branches = {**branch_dict['cand_branches'], **branch_dict['scalar_branches']}
        for br, br_name in output_branches.items():
            #check if it is a lepton
            if br in outleps.keys():
                selected_evts = outleps[br][presel_mask]
            #check if it a scalar
            elif br in scalars.keys():
                selected_evts = scalars[br][presel_mask]
            else:
                if br in copied_branches.keys():
                     selected_evts = ak.flatten(tree[br])[presel_mask]
                else:
                     selected_evts = copied_branches[br][presel_mask]

            if br_name not in tree_values.keys():
                 tree_values.update({br_name:selected_evts})
            else:
                 tree_values[br_name]=np.concatenate((tree_values[br_name], selected_evts))

    name = 'measurement_bdt' if args.mode=='measure' else 'trainBkg_bdt'
    if args.useBsideBands: name += '_sideBands'
    if args.useLowQ: name += '_lowQ'
    if args.useHighQ: name += '_highQ'
    if args.total>0: name += f'_maxFiles_{str(args.total)}'
    if args.label: name+=f'_{args.label}'

    name = name+'_part'+str(ipart) if ipart is not None else name
    with ur.recreate(args.outpath+'/'+name+'.root') as f:
        f['mytree'] = tree_values

def mp_worker(files,ipart,args,branch_dict):
    if args.mode=='split':
        args.mode='train'
        args.split='event%2==0'
        preprocess_inputs(files,ipart,args,branch_dict)
        args.mode='measure'
        args.split='event%2!=0'
        preprocess_inputs(files,ipart,args,branch_dict)
    else:
        preprocess_inputs(files,ipart,args,branch_dict)

    print(f'Part {ipart} Finished')

if __name__ == '__main__':
    parser= argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', dest='mode',default='measure',type=str)
    parser.add_argument('-j', '--nparts',dest='nparts',default=1,type=int)
    parser.add_argument('-N', '--totalfiles',dest='total',default=-1,type=int)
    parser.add_argument('-s', '--split',dest='split',default=True,type=bool)
    parser.add_argument('-i', '--inpath',dest='inpath',default='/eos/cms/store/group/phys_bphys/DiElectronX/nzipper/Skims/Data_Skims',type=str)
    parser.add_argument('-o', '--outpath',dest='outpath',default='.',type=str)
    parser.add_argument('-l', '--label',dest='label',default='',type=str)
    args=parser.parse_args()

    # parameters
    col = 'SkimBToKEE'
    args.nparts = args.nparts if args.nparts > 0 else mp.cpu_count()
    # args.total = args.total if args.total > 0 else
    args.writeMeasurement = True # Writes 'measurement' in root file name. Does nothing in the code. Essentially data is used for bkg and testing
    args.sortby = 'leppt' #options: leppt, eltype. sort leptons by pt or electron type
    args.useLowQ = False
    args.useHighQ = False
    args.useBsideBands = False

    # input file parameters
    jobFiles = preprocess_files(args.inpath, args.nparts, args.total)

    if 'KEE' in col:
        branch_dict = {
            'candidate' : col,
            'cand_branches' : {
                    col+'_mll_fullfit':'Mll',
                    col+'_fit_pt':'Bpt',
                    col+'_fit_mass':'Bmass',
                    col+'_fit_cos2D':'Bcos',
                    col+'_svprob':'Bprob',
                    col+'_fit_massErr':'BmassErr',
                    col+'_b_iso04':'Biso',
                    col+'_l_xy_sig':'BsLxy',
                    col+'_fit_l1_pt':'L1pt',
                    col+'_fit_l1_eta':'L1eta',
                    col+'_l1_iso04':'L1iso',
                    col+'_l1_pfmvaId':'L1id',
                    col+'_fit_l2_pt':'L2pt',
                    col+'_fit_l2_eta':'L2eta',
                    col+'_l2_iso04':'L2iso',
                    col+'_l2_pfmvaId':'L2id',
                    col+'_fit_k_pt':'Kpt',
                    col+'_k_iso04':'Kiso',
                    col+'_fit_k_eta':'Keta',
                    col+'_lKDz':'LKdz',
                    col+'_lKDr':'LKdr',
                    col+'_l1l2Dr':'L1L2dr',
                    col+'_k_svip3d':'Kip3d',
                    col+'_k_svip3d_err':'Kip3dErr',
                    col+'_l1_iso04_dca':'L1isoDca',
                    col+'_l2_iso04_dca':'L2isoDca',
                    col+'_k_iso04_dca':'KisoDca',
                    col+'_b_iso04_dca':'BisoDca',
                    # col+'_l1_n_isotrk_dca':'L1Nisotrk',
                    # col+'_l2_n_isotrk_dca':'L2Nisotrk',
                    # col+'_k_n_isotrk_dca':'KNisotrk',
                    col+'_k_dca_sig':'KsDca',
                    col+'_kl_massKPi':'KLmassD0',
                    col+'_p_assymetry':'Passymetry',
            },
            'leppairs_branches' : {
                    col+'_fit_l1_pt':col+'_fit_l2_pt',
                    col+'_fit_l1_eta':col+'_fit_l2_eta',
                    col+'_l1_pfmvaId':col+'_l2_pfmvaId',
                    col+'_l1_iso04':col+'_l2_iso04',
                    #col+'_l1_trk_mass':col+'_l2_trk_mass'
            },
            'scalar_branches' : {
                    'PV_npvs':'PV_npvs',
                    'event':'idx'
            },
            'presel' : {
                    col+'_svprob':0.0001,
                    col+'_fit_cos2D':0.9,
                    col+'_fit_pt':0.0,
                    col+'_l_xy_sig':2.0,
                    col+'_fit_k_pt':0.5,
                    col+'_mll_fullfit':0.0,
                    #col+'_trk_minxy2':0.000001
            },
            'leppairs_presel' : {
                    col+'_fit_l1_pt':2.0,
                    col+'_fit_l2_pt':2.0,
                    col+'_l1_pfmvaId':-1.5,
                    col+'_l2_pfmvaId':-3.0
            },
        }

    elif 'KMuMu' in col: pass
    else: raise KeyError('Pick Proper Column Name')

    if args.nparts>1:
        print(f'Distributing {args.total} Files to {args.nparts} workers...')

        procs = []
        for i, ifiles in enumerate(jobFiles):
            print(f'Submitting Part {str(i+1)}')
            proc = mp.Process(target=mp_worker, args=(ifiles,i+1,args,branch_dict))
            procs.append(proc)
            proc.start()

        for proc in procs:
            proc.join()

    else:
        runFiles = jobFiles

        if args.mode=='split':
            args.mode='train'
            args.split='event%2==0'
            preprocess_inputs(runFiles,None,args,branch_dict)
            args.mode='measure'
            args.split='event%2!=0'
            preprocess_inputs(runFiles,None,args,branch_dict)

        else: preprocess_inputs(runFiles,None,args,branch_dict)

