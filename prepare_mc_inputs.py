import time
import argparse
import ROOT as rt
from tqdm import tqdm
from array import array
from math import isinf, isnan
from glob import glob

def preprocess_inputs(runFiles,args,branch_dict):
    # Branch Parameters
    MBmin = 4.5
    MBmax = 6.0
    Mll_lowQ = 3.0 # applied only if mode is training. measurment mode ignores it
    Mll_highQ = 4.00 # applied only in highq2train
    addMassConstraintVariables = True

    if args.mode=='train':
        args.useLowQ = True
        args.writeMeasurment = False
    elif args.mode=='measure':
        args.useLowQ = False
        args.writeMeasurment = True

    name = 'trainSgn_bdt'
    if args.writeMeasurment: name = 'MCmeasurment_bdt'
    name += '_KMuMu' if args.lepton=='Mu' else ('_KEE_PFeLowPt' if args.lepton=='LowPt' else ('_KEE_PFe' if args.lepton=='PFe' else ''))
    if args.useLowQ: name+='_lowQ'
    if args.AddHighQ: name+='_highQ'
    if args.nstart!=0 or args.nend!=-1: name+='_Evts_'+str(round(args.nstart))+'-'+str(round(args.nend))
    
    if args.specificTrigger:
        print('Requiring',args.specificTrigger,' trigger')
        name+='_'+args.specificTrigger.split('_')[1]+args.specificTrigger.split('_')[2]

    assert args.sortby=='eltype' or args.sortby=='leppt'
    
    ############################################ WIP Update using Uproot ############################################
    #################################################################################################################
    
    # ncands_branch = 'nBToKMuMu' if 'Mu' in args.lepton else 'nBToKEE'
    # needed_branches = list({ncands_branch} | branch_dict['output_branches'].keys() | branch_dict['l1_branches'].keys() | branch_dict['l2_branches'].keys())
    # needed_branches = [br for br in needed_branches if 'sortedlep' not in br]
    # tree_values = {}
    # for i, tree in enumerate(ur.iterate([runFile+":Events" for runFile in runFiles],needed_branches,cut=args.split,namedecode="utf-8",library="ak")):
    #     presel_mask = np.full(tree[branch_dict['candidate']+"_fit_mass"].to_numpy().shape, True)
    #     # we want to rearrange leptons to make sure that they are properly sorted (pt or type). so the output leptonX will have two contributions from input leptonX and Y
    #     outl1_inl1_mask = np.copy(presel_mask)
    #     outl1_inl2_mask = np.copy(presel_mask)

    #     # Deal with scalars
    #     scalars = {}
    #     entries_per_evt = tree[ncands_branch]
    #     for br, outname in branch_dict['scalar_branches'].items():
    #         values_scl = tree[br]
    #         scalars[br] = np.repeat(values_scl,entries_per_evt)


    #     for br, br_name in branch_dict['output_branches'].items():
    #         selected_evts = np.ones_like(presel_mask)
    #         if False:
    #             pass
    #         # #check if it is a lepton
    #         # if br in outleps.keys():
    #         #     selected_evts = outleps[br][presel_mask]
    #         #check if it a scalar
    #         elif br in scalars.keys():
    #             selected_evts = scalars[br][presel_mask]
    #         else:
    #             # if br in copied_branches.keys():
    #             #      selected_evts = ak.flatten(tree[br])[presel_mask]
    #             # else:
    #             #      selected_evts = copied_branches[br][presel_mask]
    #             pass
    #         if br_name not in tree_values.keys():
    #             tree_values.update({br_name:selected_evts})
    #         else:
    #             tree_values[br_name]=np.concatenate((tree_values[br_name], selected_evts))
    #     with ur.recreate(args.outpath+"/"+name+"_uproot.root") as f:
    #         f["mytree"] = tree_values

    #################################################################################################################
    #################################################################################################################

    tree = rt.TChain('Events')
    for f in runFiles: 
        tree.Add(f)

    print(args.mode.capitalize()+' Mode')
    print(f'- sorting by {args.sortby}')
    if tree.GetEntries()<args.nend:
        args.nend=tree.GetEntries()
    print('- start=',round(args.nstart),'end=',round(args.nend),'all entries=',tree.GetEntries())

    if args.nstart>0 and args.nstart<1.0:
        args.nstart=int(tree.GetEntries()*args.nstart)
    if args.nend>0 and args.nend<1.1:
        args.nend=int(tree.GetEntries()*args.nend)
    if args.nend<0:
        args.nend=tree.GetEntries()

    output_tree = rt.TTree('mytree','mytree')

    output_array = {branch: array('i' if 'event' in branch else 'f',[0]) for branch in branch_dict['output_branches'].keys() }
    for branch in branch_dict['output_branches'].keys():
        output_tree.Branch(branch_dict['output_branches'][branch],output_array[branch],branch_dict['output_branches'][branch]+f'/{"I" if "event" in branch else "F"}')

    if args.addMlkVariables:
        output_array['MLK_BothMuMass'] = array('f',[0])
        output_array['MLK_KPiMass'] = array('f',[0])
        output_tree.Branch('MLK_BothMuMass',output_array['MLK_BothMuMass'],'MLK_BothMuMass/F')
        output_tree.Branch('MLK_KPiMass',output_array['MLK_KPiMass'],'MLK_KPiMass/F')

    if addMassConstraintVariables:
        output_array['Bmass_constraintJpsi'] = array('f',[0])
        output_array['Bmass_constraintPsi2S'] = array('f',[0])
        output_tree.Branch('Bmass_constraintJpsi',output_array['Bmass_constraintJpsi'],'Bmass_constraintJpsi/F')
        output_tree.Branch('Bmass_constraintPsi2S',output_array['Bmass_constraintPsi2S'],'Bmass_constraintPsi2S/F')   


    iev = 0
    tstart = time.time()
    tot = 0
    for ev in tqdm(tree, total=args.nend):
        if iev < args.nstart: 
            iev += 1
            continue
        if iev==args.nend: break
        iev+=1 
        if args.specificTrigger is not None and ord(getattr(ev,args.specificTrigger))==0: continue

        if args.mode=='train' and args.split=='event%2==0':
            if getattr(ev,'event')%2!=0: continue
        if args.mode=='measure' and args.split=='event%2!=0':
            if getattr(ev,'event')%2==0: continue

        if len (branch_dict['presel_trg'].keys())>0:
            ntrgmu=0
            for lep in ['1','2']:
                    pt=getattr(ev,'recoMu'+lep+'_pt')
                    eta=getattr(ev,'recoMu'+lep+'_eta')
                    dxyErr=getattr(ev,'recoMu'+lep+'_dxyErr')
                    dxy=getattr(ev,'recoMu'+lep+'_dxy')
                    sdxy=0
                    if dxyErr>0: 
                        sdxy=abs(dxy)/abs(dxyErr)
                    trg=getattr(ev,'recoMu'+lep+'_isTrg')
                    if pt>branch_dict['presel_trg']['pt'] and abs(eta)<branch_dict['presel_trg']['eta'] and abs(sdxy)>branch_dict['presel_trg']['sdxy'] and trg==1:
                        ntrgmu+=1
            if ntrgmu==0: continue   
        tot+=1
        # reconstruction DR
        skip=False
        for cut in branch_dict['presel_dr'].keys():
            if getattr(ev,cut)>branch_dict['presel_dr'][cut]: 
                skip=True
                break
        if skip:  continue
        #read branches for preselection
        branches={ cut:getattr(ev,cut) for cut in branch_dict['presel'].keys() }  
        skip=False
        # apply cuts
        for cut in branches.keys():
            if branches[cut]<branch_dict['presel'][cut]:
                skip=True
                break
        if skip: continue
        lep1_branches=branch_dict['l1_branches'].copy()
        lep2_branches=branch_dict['l2_branches'].copy()
        if args.lepton=='PFe' and ( getattr(ev,'recoE1_isPF') != 1  or getattr(ev,'recoE2_isPF') != 1 ):
            continue
        if args.lepton=='LowPt' and ( getattr(ev,'recoE1_isPF') + getattr(ev,'recoE2_isPF') != 1 ):
            continue

        if args.sortby=='eltype':
            if getattr(ev,'recoE1_isPF') and not getattr(ev,'recoE2_isPF') and not getattr(ev,'recoE2_isPFoverlap'):
                lep1_branches.pop(cols['L1']+'_mvaId',None)
                lep2_branches.pop(cols['L2']+'_pfmvaId',None)
            elif getattr(ev,'recoE2_isPF') and not getattr(ev,'recoE1_isPF') and not getattr(ev,'recoE1_isPFoverlap'):
                lep1_branches.pop(cols['L1']+'_pfmvaId',None)
                lep2_branches.pop(cols['L2']+'_mvaId',None)
                lep1_branches,lep2_branches = lep2_branches, lep1_branches
            else: 
                print('Warning problem sorting by type failed')
                print('e1 pf',getattr(ev,'recoE1_isPF'),'e2 pf',getattr(ev,'recoE2_isPF'),'bpt',getattr(ev,'recoB_fit_pt'))
                continue 
        else:
            for branch in lep1_branches.keys():
                if '_pt' in branch:
                    pt1 = getattr(ev,branch)
                    break
            for branch in lep2_branches.keys():
                if '_pt' in branch:
                    pt2 = getattr(ev,branch)
                    break
            if 'recoE' in cols['L1']:
                if getattr(ev,'recoE1_isPF'):
                    lep1_branches.pop(cols['L1']+'_mvaId',None)
                else:
                    lep1_branches.pop(cols['L1']+'_pfmvaId',None)
                if getattr(ev,'recoE2_isPF'):
                    lep2_branches.pop(cols['L2']+'_mvaId',None)
                else:
                    lep2_branches.pop(cols['L2']+'_pfmvaId',None)

            lep2_charge, k_charge = 0, 0
            if args.lepton=='Mu':
                lep2_charge = getattr(ev,'genMu2_charge')
    #      k_charge = getattr(ev,'genK_charge')

            if pt1<pt2:
                lep1_branches,lep2_branches = lep2_branches, lep1_branches
                

        #naming
        presel_daughter=0
        for branch in lep1_branches.keys(): 
            lep1_branches[branch]= lep1_branches[branch].format(order=str(1))
            if lep1_branches[branch] in branch_dict['presel_lep'].keys():
                if getattr(ev,branch)>branch_dict['presel_lep'][lep1_branches[branch]]:
                    presel_daughter+=1
        for branch in lep2_branches.keys():
            lep2_branches[branch]= lep2_branches[branch].format(order=str(2))
            if lep2_branches[branch] in branch_dict['presel_lep'].keys():
                if getattr(ev,branch)>branch_dict['presel_lep'][lep2_branches[branch]]:
                    presel_daughter+=1

        if presel_daughter<len(branch_dict['presel_lep']): continue
        eta1=getattr(ev,'{0}_{1}'.format(cols['L1'],'eta'))
        eta2=getattr(ev,'{0}_{1}'.format(cols['L2'],'eta'))
        etaK=getattr(ev,'{0}_{1}'.format(cols['B'],'fit_k_eta'))
        if abs(eta1)>2.4 or abs(eta2)>2.4: continue
        if abs(etaK)>2.4: continue
        if args.selectEtaBin=='BB' and ( abs(eta1)>1.5 or abs(eta2)>1.5): continue
        if args.selectEtaBin=='BE_EE' and ( abs(eta1)<1.5 and abs(eta2)<1.5): continue
        MB=getattr(ev,'{0}_{1}'.format(cols['B'],'fit_mass'))
        if MB<MBmin or MB>MBmax: continue
        Mll=getattr(ev,'{0}_{1}'.format(cols['B'],'mll_fullfit'))
        if args.useLowQ and (not args.AddHighQ) and Mll > Mll_lowQ: 
            continue
        if args.useLowQ and args.AddHighQ and (Mll > Mll_lowQ and Mll<Mll_highQ): 
            continue
        MLK_values={}
        if args.addMlkVariables:
            vlep = rt.TLorentzVector()
            vK = rt.TLorentzVector()
            if lep2_charge==k_charge:
                    vlep.SetPtEtaPhiM(getattr(ev,'{0}_fit_l1_pt'.format(cols['B'])), getattr(ev,'{0}_fit_l1_eta'.format(cols['B'])),getattr(ev,'{0}_fit_l1_phi'.format(cols['B'])),0.105)
            else:
                    vlep.SetPtEtaPhiM(getattr(ev,'{0}_fit_l2_pt'.format(cols['B'])), getattr(ev,'{0}_fit_l2_eta'.format(cols['B'])),getattr(ev,'{0}_fit_l2_phi'.format(cols['B'])),0.105)
            vK.SetPtEtaPhiM(getattr(ev,'recoB_fit_k_pt'), getattr(ev,'recoB_fit_k_eta'),getattr(ev,'recoB_fit_k_phi'),0.105)
                        
            MLK_values['MLK_BothMuMass']=(vK+vlep).M()
            vlep.SetPtEtaPhiM(vlep.Pt(),vlep.Eta(),vlep.Phi(),0.493)
            vK.SetPtEtaPhiM(vK.Pt(),vK.Eta(),vK.Phi(),0.139)
            MLK_values['MLK_KPiMass']=(vK+vlep).M()

        constraintMass={}
        if addMassConstraintVariables:
            constraintMass['Bmass_constraintJpsi']=MB-Mll+3.097
            constraintMass['Bmass_constraintPsi2S']=MB-Mll+3.686
            
        inf_nan_veto=False
        sortedlep={}
        for key in lep1_branches:
            sortedlep[lep1_branches[key]] = getattr(ev,key)
            if isinf(getattr(ev,key)) or isnan(getattr(ev,key)):
                inf_nan_veto=True
        for key in lep2_branches:
            sortedlep[lep2_branches[key]] = getattr(ev,key)
            if isinf(getattr(ev,key)) or isnan(getattr(ev,key)):
                inf_nan_veto=True
        for key in output_array.keys():
            if key==cols['B']+'_mll_fullfit': output_array[key][0]=Mll
            elif key==cols['B']+'_fit_mass': output_array[key][0]=MB
            elif key in branches.keys():
                output_array[key][0]=branches[key]
                if isinf(branches[key]) or isnan(branches[key]):
                        inf_nan_veto=True
            elif key in sortedlep.keys():
                output_array[key][0]=sortedlep[key]
            elif key in MLK_values.keys():
                output_array[key][0]=MLK_values[key]
            elif key in constraintMass.keys():
                output_array[key][0]=constraintMass[key]

            else:
                if 'HLT' in key:
                    if ord(getattr(ev,key)): 
                            output_array[key][0]=1.
                    else: 
                            output_array[key][0]=0.
                else:
                    output_array[key][0]=getattr(ev,key)
                    if isinf(getattr(ev,key)) or isnan(getattr(ev,key)): 
                            inf_nan_veto=True
        if inf_nan_veto:
            continue
        output_tree.Fill()

    name+='_'+args.label if args.label else ''
    with rt.TFile(f'{args.outpath}/{name}.root','RECREATE'):
        output_tree.Write()
    print(f'{tot} Events Processed')
    print(f'{round(time.time()-tstart)} Seconds Elapsed')


if __name__ == '__main__':
    parser= argparse.ArgumentParser()
    parser.add_argument('--nstart', dest='nstart', default=0, type=float)
    parser.add_argument('--nend', dest='nend', default=-1, type=float)
    parser.add_argument('-m', '--mode', dest='mode', default='measure', type=str)
    parser.add_argument('-s', '--split', dest='split', default=True, type=str)
    parser.add_argument('-N', '--totalfiles', dest='total', default=-1, type=int)
    parser.add_argument('--lepton', dest='lepton', default='PFe', choices=['PFe', 'Mu', 'LowPt'], type=str, help='choices reco lepton options: Mu, PFe, or LowPt')
    parser.add_argument('-l', '--label',dest='label',default='',type=str)
    parser.add_argument('--addHighQtrain', dest='AddHighQ', default=False, action='store_true')
    parser.add_argument('-i', '--inpath',dest='inpath',default='/eos/cms/store/group/phys_bphys/DiElectronX/nzipper/Skims/MC_Rare_Skims',type=str)
    parser.add_argument('-o', '--outpath',dest='outpath',default='.',type=str)
    args=parser.parse_args()

    # read inputs
    files = [args.inpath] if args.inpath.endswith('.root') else glob(args.inpath+'/**/*.root',recursive=True)[:args.total]
    # files = glob(filedir+'/**/*.root',recursive=True)[:args.total]

    # parameters
    args.writeMeasurment = True #prints measurment in the final root file. Does nothing in the code. Essentially data is used for bkg and testing
    args.useLowQ = False
    args.specificTrigger = None # options: None (without quotes): not requiring specific path or HLT path
    args.selectEtaBin = None # options: None (without quotes), BB, BE_EE
    args.sortby = 'leppt' #two options eltype (1 ->PF 2->low) or leppt (1 -> leading, 2->subleading) use first for kmumu kee (with 2pf ) and the second for kee (low + pf)
    args.addMlkVariables = False #for now only for muons... Also takes gen charge for the opposite sign pair


    #select reco lepton flavour
    cols = {
        'B':'recoB',
        'L1':f'reco{"Mu" if "Mu" in args.lepton else "E"}1',
        'L2':f'reco{"Mu" if "Mu" in args.lepton else "E"}2',
        'K':'recoK',
    }
    if True:
        branch_dict = {
            'candidate' : cols['B'],
            'output_branches' : {
                    cols['B']+'_mll_fullfit':'Mll',
                    cols['B']+'_fit_pt':'Bpt',
                    cols['B']+'_fit_mass':'Bmass',
                    cols['B']+'_fit_cos2D':'Bcos',
                    cols['B']+'_svprob':'Bprob',
                    cols['B']+'_fit_massErr':'BmassErr',
                    cols['B']+'_b_iso04':'Biso',
                    cols['B']+'_l_xy_sig':'BsLxy',
                    cols['B']+'_lKDz':'LKdz',
                    cols['B']+'_lKDr':'LKdr',
                    cols['B']+'_l1l2Dr':'L1L2dr',
                    cols['B']+'_fit_k_pt':'Kpt',
                    cols['B']+'_k_iso04':'Kiso',
                    cols['B']+'_fit_k_eta':'Keta',
                    cols['B']+'_k_svip3d':'Kip3d',
                    cols['B']+'_k_svip3d_err':'Kip3dErr',
                    cols['B']+'_k_iso04_dca':'KisoDca',
                    cols['B']+'_b_iso04_dca':'BisoDca',
                    cols['B']+'_l1_n_isotrk_dca':'L1Nisotrk',
                    cols['B']+'_l2_n_isotrk_dca':'L2Nisotrk',
                    cols['B']+'_k_n_isotrk_dca':'KNisotrk',
                    cols['K']+'_DCASig':'KsDca',
                    cols['B']+'_k_opp_l_mass':'KLmassD0',
                    # cols['B']+'_trk_minxy1':'BTrkdxy1'
                    # cols['B']+'_trk_minxy2':'BTrkdxy2',
                    # cols['B']+'_trk_minxy3':'BTrkdxy3',
                    # cols['B']+'_trk_mean':'BTrkMean',
                    'sortedlep1_pt':'L1pt','sortedlep2_pt':'L2pt',
                    'sortedlep1_eta':'L1eta','sortedlep2_eta':'L2eta',
                    'sortedlep1_id':'L1id','sortedlep2_id':'L2id',
                    'sortedlep1_iso':'L1iso','sortedlep2_iso':'L2iso',
                    'sortedlep1_iso04_dca':'L1isoDca','sortedlep2_iso04_dca':'L2isoDca',
                    # 'sortedlep1_trk_mass':'L1Trkmass','sortedlep2_trk_mass':'L2Trkmass',
                    cols['B']+'_p_assymetry':'Passymetry',
                    'PV_npvs':'Npv',
                    'event':'idx',
            },
            'scalar_branches' : {
                    'nBToKEE':'nBToKEE',
                    'PV_npvs':'Npv',
                    'event':'idx',
            },
            'l1_branches' : {
                    cols['B']+'_fit_l1_pt':'sortedlep{order}_pt',
                    cols['B']+'_fit_l1_eta':'sortedlep{order}_eta',
                    cols['B']+'_l1_iso04':'sortedlep{order}_iso',
                    cols['L1']+'_PFEleMvaID_RetrainedRawValue':'sortedlep{order}_id',
                    # cols['B']+'_l1_trk_mass':'sortedlep{order}_trk_mass',
                    cols['B']+'_l1_iso04_dca':'sortedlep{order}_iso04_dca',
            },
            'l2_branches' : {
                    cols['B']+'_fit_l2_pt':'sortedlep{order}_pt',
                    cols['B']+'_fit_l2_eta':'sortedlep{order}_eta',
                    cols['B']+'_l2_iso04':'sortedlep{order}_iso',
                    cols['L2']+'_PFEleMvaID_RetrainedRawValue':'sortedlep{order}_id',
                    # cols['B']+'_l2_trk_mass':'sortedlep{order}_trk_mass',
                    cols['B']+'_l2_iso04_dca':'sortedlep{order}_iso04_dca',
            },
            'presel' : {
                    # cols['B']+'_svprob':0.0001,
                    # cols['B']+'_fit_cos2D':0.9,
                    # cols['B']+'_fit_pt':0.0,
                    # cols['B']+'_l_xy_sig':3.0,
                    # cols['K']+'_pt':0.5,
                    # cols['B']+'_mll_fullfit':1.05,
                    # cols['B']+'_trk_minxy2':0.000001,
            },
            'presel_lep' :  {
                    # 'sortedlep1_pt':2.0,
                    # 'sortedlep2_pt':2.0,
                    # 'sortedlep1_id':-1.5,
                    # 'sortedlep2_id':-3.0
            },
            'presel_dr' : {},
            # 'presel_dr' : {cols['L1']+'_DR':0.03,cols['L2']+'_DR':0.03,cols['K']+'_DR':0.03},
            'presel_trg' : {},
        }

    elif 'Mu' in cols['L1']: pass
    else: raise KeyError('Pick Proper Column Name')

    if args.mode=='split':
        args.mode='train'
        args.split='event%2==0'
        preprocess_inputs(files, args, branch_dict)
        args.mode='measure'
        args.split='event%2!=0'
        preprocess_inputs(files, args, branch_dict)
    else:
        preprocess_inputs(files, args, branch_dict)

