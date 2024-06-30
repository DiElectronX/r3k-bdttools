import uproot
import pandas as pd

# Step 1: Read the ROOT file
file_path = "/eos/user/j/jodedra/AnalysisWork_2024/NoahBDTtools/r3k-bdttools/outputpreparedinclusivemc26_06_24/inclusivetotalfixed.root"
with uproot.open(file_path + ":mytree") as file:  # Assuming the tree is named 'mytree'
    i=0
    for batch in file.iterate(step_size="10mb", library='pd'):
        print(i)
        batch["trig_wgt"] = 1
        with uproot.recreate("/eos/user/j/jodedra/AnalysisWork_2024/NoahBDTtools/r3k-bdttools/withtrig_wgts/haddertrigwgts_"+str(i)+"_.root") as new_file:
            new_file["mytree"] = batch
        i = i + 1
