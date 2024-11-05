import uproot
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt

bdt_input_files = {
    'k_jpsi_kaon'      : '/Users/noahzipper/Desktop/Research/Rk/BDT_Studies/RootFiles/xval_jpsi_v2.root',
    'kstar_jpsi_kaon'  : '/Users/noahzipper/Desktop/Research/Rk/BDT_Studies/RootFiles/measurement_kstar_jpsi_kaon.root',
    'kstar_jpsi_pion'  : '/Users/noahzipper/Desktop/Research/Rk/BDT_Studies/RootFiles/measurement_kstar_jpsi_pion.root',
    'k0star_jpsi_kaon' : '/Users/noahzipper/Desktop/Research/Rk/BDT_Studies/RootFiles/measurement_k0star_jpsi_kaon.root',
    'k0star_jpsi_pion' : '/Users/noahzipper/Desktop/Research/Rk/BDT_Studies/RootFiles/measurement_k0star_jpsi_pion.root',
    'chic1_jpsi_kaon'  : '/Users/noahzipper/Desktop/Research/Rk/BDT_Studies/RootFiles/measurement_chic1_jpsi_kaon.root',
    'jpsipi_jpsi_pion' : '/Users/noahzipper/Desktop/Research/Rk/BDT_Studies/RootFiles/measurement_jpsipi_jpsi_pion.root',
}

bdt_output_files = {
    'k_jpsi_kaon'      : '/Users/noahzipper/Desktop/Research/Rk/BDT_Studies/r3k-bdttools/outputs/10_30_24/measurement_jpsi.root',
    'kstar_jpsi_kaon'  : '/Users/noahzipper/Desktop/Research/Rk/BDT_Studies/r3k-bdttools/outputs/10_30_24/measurement_kstar_jpsi_pion.root',
    'kstar_jpsi_pion'  : '/Users/noahzipper/Desktop/Research/Rk/BDT_Studies/r3k-bdttools/outputs/10_30_24/measurement_kstar_jpsi_kaon.root',
    'k0star_jpsi_kaon' : '/Users/noahzipper/Desktop/Research/Rk/BDT_Studies/r3k-bdttools/outputs/10_30_24/measurement_k0star_jpsi_pion.root',
    'k0star_jpsi_pion' : '/Users/noahzipper/Desktop/Research/Rk/BDT_Studies/r3k-bdttools/outputs/10_30_24/measurement_k0star_jpsi_kaon.root',
    'chic1_jpsi_kaon'  : '/Users/noahzipper/Desktop/Research/Rk/BDT_Studies/r3k-bdttools/outputs/10_30_24/measurement_chic1_jpsi_kaon.root',
    'jpsipi_jpsi_pion' : '/Users/noahzipper/Desktop/Research/Rk/BDT_Studies/r3k-bdttools/outputs/10_30_24/measurement_jpsipi_jpsi_pion.root',
}

process_labels = {
    'k_jpsi_kaon'      : r'$B^{\pm} \rightarrow J/\psi(\rightarrow e^{\pm}e^{\mp}) K^{\pm}$ (Kee Candidate)',
    'kstar_jpsi_kaon'  : r'$B^{\pm} \rightarrow J/\psi(\rightarrow e^{\pm}e^{\mp}) K^{*\pm}(\rightarrow K^{\pm}\pi^{0})$ (Kee Candidate)',
    'kstar_jpsi_pion'  : r'$B^{\pm} \rightarrow J/\psi(\rightarrow e^{\pm}e^{\mp}) K^{*\pm}(\rightarrow K_{S}^{0}\pi^{\pm})$ ($\pi$ee Candidate)',
    'k0star_jpsi_kaon' : r'$B^{\pm} \rightarrow J/\psi(\rightarrow e^{\pm}e^{\mp}) K^{*0}(\rightarrow K^{\pm}\pi^{\pm})$ (Kee Candidate)',
    'k0star_jpsi_pion' : r'$B^{\pm} \rightarrow J/\psi(\rightarrow e^{\pm}e^{\mp}) K^{*0}(\rightarrow K^{\pm}\pi^{\pm})$ ($\pi$ee Candidate)',
    'chic1_jpsi_kaon'  : r'$B^{\pm} \rightarrow \chi_{c1}(\rightarrow J/\psi(\rightarrow e^{\pm}e^{\mp})) K^{\pm}$ (Kee Candidate)',
    'jpsipi_jpsi_pion' : r'$B^{\pm} \rightarrow J/\psi(\rightarrow e^{\pm}e^{\mp}) \pi^{\pm}$ ($\pi$ee Candidate)',
}

plot_cfgs = {
    'Bprob'          : {'norm' : True, 'logy' : False, 'bins' : np.linspace(0,1,25),},
    'BsLxy'          : {'norm' : True, 'logy' : False, 'bins' : np.linspace(0,250,50),},
    'L2iso/L2pt'     : {'norm' : True, 'logy' : True, 'bins' : np.linspace(0,100,50),},
    'Bcos'           : {'norm' : True, 'logy' : True, 'bins' : np.linspace(.9,1,50),},
    'Kiso/Kpt'       : {'norm' : True, 'logy' : False, 'bins' : np.linspace(0,60,50),},
    'LKdz'           : {'norm' : True, 'logy' : False, 'bins' : np.linspace(0,1,50),},
    'LKdr'           : {'norm' : True, 'logy' : False, 'bins' : np.linspace(0,3,50),},
    'Passymetry'     : {'norm' : True, 'logy' : False, 'bins' : np.linspace(-1,1,50),},
    'Kip3d/Kip3dErr' : {'norm' : True, 'logy' : False, 'bins' : np.linspace(-6,6,50),},
    'L1id'           : {'norm' : True, 'logy' : False, 'bins' : np.linspace(-6,6,25),},
    'L2id'           : {'norm' : True, 'logy' : False, 'bins' : np.linspace(-6,6,25),},
    'L1iso'          : {'norm' : True, 'logy' : False, 'bins' : np.linspace(0,100,50),},
    'L2iso'          : {'norm' : True, 'logy' : False, 'bins' : np.linspace(0,100,50),},
    'bdt_score'      : {'norm' : True, 'logy' : False, 'bins' : np.linspace(-20,20,50),},
    'default'        : {'norm' : True, 'logy' : False, 'bins' : np.linspace(0,100,50),},
}

def apply_cut_and_plot(input_files, branches, cut=None, selected_keys=None):
    for branch in branches:
        plot_cfg = plot_cfgs[branch] if branch in plot_cfgs else plot_cfgs['default']
        fig, ax = plt.subplots(figsize=(8, 6), layout='constrained')
        bin_centers = None
        filekeys = input_files.keys() if selected_keys is None else selected_keys
        for key in filekeys:
            file = input_files[key]
            with uproot.open(file) as f:
                tree = f["mytree"]
                arrays = tree.arrays([branch, 'trig_wgt'], cut)
                norm = arrays['trig_wgt'] / ak.sum(arrays['trig_wgt']) if plot_cfg['norm'] else arrays['trig_wgt']
                hist, bin_edges = np.histogram(arrays[branch], bins=plot_cfg['bins'], weights=norm)
                bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1]) if bin_centers is None else bin_centers
                
                ax.errorbar(bin_centers, hist, yerr=0, marker='', drawstyle='steps-mid', label=process_labels[key])
        
        ax.set_xlabel(branch,loc='right')
        ax.set_ylabel(r'$N_{Candidates}$', loc='top')
        ax.set_yscale('log' if plot_cfg['logy'] else 'linear')
        ax.set_title(f'{branch} distribution')
        ax.legend()
        fig.savefig(f'bdt_input_plots/{branch.replace("/", "over")}_distribution.png')
        plt.close(fig)

bdt_feature_branches = ['Bprob', 'BsLxy', 'L2iso/L2pt', 'Bcos', 'Kiso/Kpt', 'LKdz', 'LKdr', 'Passymetry', 'Kip3d/Kip3dErr', 'L1id', 'L2id', 'L1iso', 'L2iso']
score_branch = ['bdt_score']

cut_string = 'KLmassD0 > 2.'

selected_keys = ['kstar_jpsi_kaon','k0star_jpsi_kaon']
apply_cut_and_plot(bdt_input_files, bdt_feature_branches, cut=cut_string, selected_keys=None)

selected_keys = ['k_jpsi_kaon', 'jpsipi_jpsi_pion']
apply_cut_and_plot(bdt_output_files, score_branch, selected_keys=None)
