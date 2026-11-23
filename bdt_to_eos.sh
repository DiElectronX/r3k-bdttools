#!/usr/bin/env bash
#
# bdt_to_eos.sh -- push BDT inference outputs from local scratch to EOS, with
# integrity verification, in one step. Replaces the old "cp to EOS + run
# rename_bdt_output.sh" flow.
#
# WHY
#   Writing ROOT files through the /eos FUSE mount (either a ROOT/uproot write or a
#   plain `cp`) silently corrupts baskets: the file opens and reports an entry count,
#   but reading entries segfaults. This script avoids that two ways:
#     1) it reads the LOCAL source end-to-end with uproot first (streams every branch
#        so every basket decompresses) -- catches a source that was born corrupt;
#     2) it copies with `xrdcp -f -p -C adler32:source` over XRootD (NOT FUSE), which
#        verifies the landed EOS file's adler32 against the source and FAILS the copy
#        on any mismatch -- catches transfer/storage corruption.
#   (1)+(2) together guarantee the remote file is good without needing uproot to open
#   root:// URLs. Any file that fails is reported and the script exits non-zero, so it
#   never leaves a silent landmine for the fitter.
#
# USAGE
#   ./bdt_to_eos.sh [LOCAL_SRC_DIR] [EOS_DEST_BASE]
#     LOCAL_SRC_DIR  dir holding measurement_*.root  (default: /tmp/jfratice/bdt_out)
#     EOS_DEST_BASE  XRootD URL of the test_output_mix dir
#                    (default: run4_splot_jpsi/.../test_output_mix)
#
#   Run it in an environment that has `xrdcp` and python with `uproot`
#   (e.g. `conda activate r3k-bdt`).
#
set -uo pipefail

SRC="${1:-/tmp/jfratice/bdt_out}"
EOS_BASE="${2:-root://eoscms.cern.ch//eos/cms/store/group/phys_bphys/DiElectronX/jfratice/r3k-analysis/output/r3k-int/run4_splot_jpsi/data/test_output_mix}"
TREE="Events"

# measurement file (in SRC)            -> destination relative path (under EOS_BASE)
MAP=(
  "measurement_chic1_jpsi_kaon.root|jpsi/Bu_kaon_chic1_jpsi_kaon_skimmed_mix_wScores.root"
  "measurement_data.root|data/data2022_skimmed_mix_wScores.root"
  "measurement_jpsi_jpsi.root|jpsi/Bu_kaon_jpsi_resonant_skimmed_mix_wScores.root"
  "measurement_jpsi_lowq2.root|lowq2/Bu_kaon_jpsi_resonant_skimmed_mix_wScores.root"
  "measurement_jpsipi_jpsi_pion.root|jpsi/Bu_jpsi_pi_pion_skimmed_mix_wScores.root"
  "measurement_k0star_jpsi_kaon.root|jpsi/Bd_k0star_jpsi_kaon_pion_kaon_skimmed_mix_wScores.root"
  "measurement_k0star_jpsi_pion.root|jpsi/Bd_k0star_jpsi_kaon_pion_pion_skimmed_mix_wScores.root"
  "measurement_k0star_kaon.root|lowq2/Bd_k0star_kaon_pi_kaon_skimmed_mix_wScores.root"
  "measurement_k0star_pion.root|lowq2/Bd_k0star_kaon_pi_pion_skimmed_mix_wScores.root"
  "measurement_k0star_psi2s_kaon.root|psi2s/Bd_k0star_psi2s_kaon_pion_kaon_skimmed_mix_wScores.root"
  "measurement_k0star_psi2s_pion.root|psi2s/Bd_k0star_psi2s_kaon_pion_pion_skimmed_mix_wScores.root"
  "measurement_kstar_jpsi_kaon.root|jpsi/Bu_kstar_jpsi_kaon_pi0_kaon_skimmed_mix_wScores.root"
  "measurement_kstar_jpsi_pion.root|jpsi/Bu_kstar_jpsi_ks0_pi_pion_skimmed_mix_wScores.root"
  "measurement_kstar_pion.root|lowq2/Bu_kstar_K0S_pi_pion_skimmed_mix_wScores.root"
  "measurement_kstar_psi2s_pion.root|psi2s/Bu_kstar_psi2s_KS0_pi_pion_skimmed_mix_wScores.root"
  "measurement_psi2s.root|psi2s/Bu_kaon_psi2s_resonant_skimmed_mix_wScores.root"
  "measurement_rare.root|lowq2/Bu_kaon_lowq_resonant_skimmed_mix_wScores.root"
  "measurement_same_sign_electrons.root|data/ssdata2022_skimmed_mix_wScores.root"
)

command -v xrdcp >/dev/null || { echo "ERROR: xrdcp not found (need an XRootD client; e.g. conda activate r3k-bdt)"; exit 2; }

# Streams every branch of the LOCAL file through uproot in chunks -> forces all
# baskets to decompress. Exit 0 only if the whole tree reads; non-zero otherwise.
verify_local() {
  local path="$1"
  python3 - "$path" "$TREE" <<'PY'
import sys
path, tree = sys.argv[1], sys.argv[2]
try:
    import uproot
    n = 0
    with uproot.open(path) as f:
        t = f[tree]
        total = t.num_entries
        for chunk in t.iterate(library="np", step_size=200_000):
            k = next(iter(chunk)) if chunk else None
            n += len(chunk[k]) if k is not None else 0
    if n != total:
        print(f"   local read: BAD (read {n} of {total})"); sys.exit(1)
    print(f"   local read: OK n={n}"); sys.exit(0)
except Exception as e:
    print(f"   local read: BAD ({type(e).__name__}: {e})"); sys.exit(1)
PY
}

fail=0; sent=0; missing=0
for entry in "${MAP[@]}"; do
    src_name="${entry%%|*}"
    rel_dst="${entry##*|}"
    src="$SRC/$src_name"
    dst="$EOS_BASE/$rel_dst"

    if [[ ! -e "$src" ]]; then
        echo "-- skip (not produced this run): $src_name"
        missing=$((missing+1)); continue
    fi

    echo ">> $src_name -> $rel_dst"

    # (1) source must read end-to-end before we ship it.
    if ! verify_local "$src"; then
        echo "   ^ LOCAL source is corrupt; re-run the BDT for this sample. Not shipped."
        fail=$((fail+1)); continue
    fi

    # (2) copy over XRootD with adler32 verification of the landed EOS file.
    # NOTE: XRootD 5 xrdcp does not honor the POSIX "--" end-of-options marker
    # (it discards the file args -> "No files specified"); use long flags, no "--".
    if xrdcp --force --posc --cksum adler32:source "$src" "$dst"; then
        sent=$((sent+1))
    else
        echo "   ^ xrdcp/checksum FAILED (transfer corruption or EOS error); re-run. Not usable."
        fail=$((fail+1))
    fi
done

echo "----------------------------------------------------------------"
echo "transferred+verified: $sent   skipped(missing): $missing   FAILED: $fail"
if [[ "$fail" -ne 0 ]]; then
    echo "RESULT: FAILURE -- $fail file(s) failed transfer/verification (see above)."
    exit 1
fi
echo "RESULT: OK -- all present measurement files are on EOS and read clean."
