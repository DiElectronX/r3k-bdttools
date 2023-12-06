#!/bin/bash

# Check if a directory argument is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

directory="$1"

for file in "$directory"/*.pkl; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        echo "Processing file: $file"
        python measure_bdt.py --model $file --measurefile ../RootFiles/measurement_bdt.root
        python measure_bdt.py --model $file --measurefile ../RootFiles/measurement_bdt_SameSignElectrons.root --label SameSignElectrons

        if [[ ! $filename == *"lowSideband"* ]]; then
            python measure_bdt.py --model $file --measurefile ../RootFiles/MCmeasurment_bdt_KEE_PFe_noPresel.root --label rare_noPresel
            python measure_bdt.py --model $file --measurefile ../RootFiles/MCmeasurment_bdt_JPsiK_PFe_noPresel.root --label jpsi_noPresel
        fi
    fi
done