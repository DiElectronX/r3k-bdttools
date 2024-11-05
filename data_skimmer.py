import ROOT
import sys
from os import path

def main(bdt_cut=0., bmass_range_cut=(4.5, 6)):
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_root_file>")
        sys.exit(1)

    root_file_path = sys.argv[1]

    rdf = ROOT.RDataFrame("mytree", root_file_path)

    xgb_cut = "bdt_score >= {}".format(bdt_cut)
    bmass_cut = "(Bmass >= {}) && (Bmass <= {})".format(bmass_range_cut[0], bmass_range_cut[1])

    rdf_filtered = rdf.Filter(xgb_cut).Filter(bmass_cut)

    directory, filename = path.split(root_file_path)
    base, extension = path.splitext(filename)
    new_filename = "{}_{}{}".format(base, 'slimmed', extension)
    updated_path = path.join(directory, new_filename)

    rdf_filtered.Snapshot("mytree", updated_path)

if __name__ == "__main__":
    ROOT.EnableImplicitMT()

    bdt_cut = 0.
    bmass_range_cut = (4.5, 6.)

    main(bdt_cut, bmass_range_cut)
