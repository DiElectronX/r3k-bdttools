#!/bin/bash

# Directory variables
dir="/eos/cms/store/group/phys_bphys/DiElectronX/nzipper/postCMGToolsAndBDTScoreAdder/new_trigger_mc/allfiles/"
output_dir="mc_newmethod_with_presel_25_08_25"

# List of names 
#names=("B0_k0star_jpsi_kaon_MC_.root"  "B0_k0star_jpsi_pion_MC_.root"  "B0_k0star_kaon_MC_.root"  "B0_k0star_pion_MC_.root"  "B0_k0star_psi2s_kaon_MC_.root"  "B0_k0star_psi2s_pion_MC_.root"  "B0_psi2s_kplus_piminus_kaon_MC_.root"  "B0_psi2s_kplus_piminus_pion_MC_.root"  "Bu_chic1_jpsi_kaon_kaon_MC_.root"  "Bu_jpsi_pion_pion_MC_.root"  "Bu_kstar_ee_MC_.root"  "Bu_kstar_jpsi_kaon_MC_.root"  "Bu_kstar_psi2s_kaon_MC_.root"  "Bu_psi2s_pi_pion_MC_.root")
#names=("KEENOCUT_MC_.root" "KJPSI_NO_CUT_2022_MC_.root" "KPSI2S_NO_CUT_2022_MC_.root")

names=("Bd_k0star_jpsi_kaon_pion_kaon_MC__skimmed_skimmed.root" "Bd_k0star_jpsi_kaon_pion_pion_MC__skimmed_skimmed.root" "Bd_k0star_kaon_pi_kaon_MC__skimmed_skimmed.root" "Bd_k0star_kaon_pi_pion_MC__skimmed_skimmed.root" "Bd_k0star_psi2s_kaon_pion_kaon_MC__skimmed_skimmed.root" "Bd_k0star_psi2s_kaon_pion_pion_MC__skimmed_skimmed.root" "Bu_jpsi_pi_pion_MC__skimmed_skimmed.root" "Bu_kaon_chic1_jpsi_kaon_MC__skimmed_skimmed.root" "Bu_kaon_jpsi_resonant_MC__skimmed_skimmed.root" "Bu_kaon_lowq_resonant_MC__skimmed_skimmed.root" "Bu_kaon_psi2s_resonant_MC__skimmed_skimmed.root" "Bu_kstar_jpsi_kaon_pi0_kaon_MC__skimmed_skimmed.root" "Bu_kstar_jpsi_ks0_pi_pion_MC__skimmed_skimmed.root" "Bu_kstar_K0S_pi_pion_MC__skimmed_skimmed.root" "Bu_kstar_psi2s_KS0_pi_pion_MC__skimmed_skimmed.root")





# Loop through each name
for name in "${names[@]}"; do
  echo "Processing $name..."
  filename=$(basename "$name" .root)
  echo "processing $filename"
  # Run the command using the name, dir, and output_dir variables
  python prepare_inputs.py -m measure -j 1 -i "${dir}/${name}" -o "$output_dir" -c rare -l "$filename"
  
  # Wait for the command to complete
  wait
  
  echo "$name processed."
done



