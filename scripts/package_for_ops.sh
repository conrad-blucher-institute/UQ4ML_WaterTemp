#!/usr/bin/env bash
#
# Copy the chosen phase-5 model per lead time into results/phase5_final/ops_bundle.
#
#   bash scripts/package_for_ops.sh
#
# Reads results/phase5_final/final_selections_phase5_repNum.csv and uses two of
# its columns: 'leadtime' and 'RepNum' (which trained repetition you picked
# after the COVE outlier check). The other columns are yours to keep notes in.
#
# Run it from the repo root.

# -e stop on the first failing command (a RepNum with no matching .keras makes
#    cp fail, which halts the script instead of silently shipping 20 of 21)
# -u error on an undefined variable   -o pipefail a pipeline fails if any stage does
set -euo pipefail

cd results/phase5_final
# Rebuild from scratch. The models get RENAMED on the way in, so a stale file
# from a previous run (an old RepNum) would not be overwritten -- it would just
# sit there and ship.
rm -rf ops_bundle
mkdir -p ops_bundle/ESB ops_bundle/metadata

# ---------------------------------------------------------------------------
# The awk line, one piece at a time:
#
#   -F,                       split each line into fields on commas
#   { sub(/\r$/, "") }        runs on EVERY line: strips the carriage return
#                             Excel leaves at the end of each line. Without it
#                             the last field is "1\r" and every glob below
#                             misses -- and you cannot see it in a text editor.
#                             Assigning to $0 makes awk re-split the fields.
#   NR==1 { ... ; next }      NR is the line number, so this is the header row.
#                             Walk the field names to find which column is
#                             called RepNum, remember it in `col`, then `next`
#                             skips to the following line without printing.
#                             Looking the column up BY NAME means a trailing
#                             comma or a reordered/extra column cannot silently
#                             shift which value gets read -- $NF would grab the
#                             empty field after the trailing comma.
#   $1 != "" && $col != ""    data rows only: skip blanks and any lead time
#                             whose RepNum has not been filled in yet.
#   print $1, $col            emit "leadtime repnum", one pair per line, which
#                             the while loop below reads into two variables.
# ---------------------------------------------------------------------------
awk -F, '
    { sub(/\r$/, "") }
    NR == 1 {
        for (i = 1; i <= NF; i++) if ($i == "RepNum") col = i
        if (!col) { print "no RepNum column in header" > "/dev/stderr"; exit 1 }
        next
    }
    $1 != "" && $col != "" { print $1, $col }
' final_selections_phase5_repNum.csv |
while read -r lt run; do
    # The '.keras' / '_metadata.json' suffixes are part of the pattern on
    # purpose: globbing 'run${run}*' would also match run10 when run is 1.
    # One folder per lead time, one model inside, named the way ops expects:
    #   ESB/ESB_48hr/ESB_48hr_member1.keras
    # 'member1' because we ship a single model per lead time; if ops ever takes
    # an ensemble, that suffix is where the member index goes.
    dest="ops_bundle/ESB/ESB_${lt}hr"
    mkdir -p "$dest"

    # The '.keras' suffix is part of the pattern on purpose: globbing
    # 'run${run}*' would also match run10 when run is 1.
    cp keras_files/MAPE_"${lt}"h_cycle1_*_run"${run}".keras \
       "$dest/ESB_${lt}hr_member1.keras"

    # The sidecar carries the input COLUMN ORDER, which the model cannot check
    # for itself, so it must not be lost. It is kept out of the ESB tree
    # so that tree stays exactly one .keras per folder, and keeps its ORIGINAL
    # name -- that name is the only remaining record of which repetition this
    # is, since the copy above renames it away.
    cp keras_files/MAPE_"${lt}"h_cycle1_*_run"${run}"_metadata.json \
       ops_bundle/metadata/

    echo "  LT ${lt} -> run ${run}  ->  ESB_${lt}hr/ESB_${lt}hr_member1.keras"
done

# Keep the sheet with the models: it records which rep you picked and why.
cp final_selections_phase5_repNum.csv ops_bundle/

# -c create  -z gzip  -f <file>
# -C cd's there first, so paths inside the archive start at ops_bundle/ instead
#    of embedding C:/Users/hmarrero/... for whoever unpacks it.
ARCHIVE="esb_ops_models_$(date +%Y%m%d).tar.gz"
tar -czf "$ARCHIVE" -C . ops_bundle

echo
echo "models : $(find ops_bundle/ESB -name '*.keras' | wc -l)"
echo "archive: $PWD/$ARCHIVE  ($(du -h "$ARCHIVE" | cut -f1))"
