#!/bin/sh
# End-to-end PrefPairs demo on the bundled sample: a seed-0 simulation with
# known ground truth (6 models, 12 annotators, 4 of them planted as defective).
# It imports the JSONL, summarises it, ranks the models with Bradley-Terry and
# with Elo (leaving the planted annotators out), runs the annotator checks, and
# fails unless both rankings recover the true model order and the checks flag
# exactly the planted position-biased and length-biased annotators.
#
#   PREFPAIRS  command that runs the CLI (default: prefpairs)
#   DEMO_DB    database to (re)create      (default: .prefpairs/demo.db)
#   EXAMPLES   directory of the sample     (default: examples)
set -eu

PREFPAIRS="${PREFPAIRS:-prefpairs}"
DB="${DEMO_DB:-.prefpairs/demo.db}"
EXAMPLES="${EXAMPLES:-examples}"
TRUTH="$EXAMPLES/sample-truth.json"
EXPECT="Kendall tau against the true order: 1.000"
EXPECT_FLAGS="flagged: ann-10 (position), ann-11 (length)"

step() { printf '\n$ prefpairs %s\n' "$*"; }

rm -f "$DB"
step import "$EXAMPLES/sample.jsonl"
$PREFPAIRS import "$EXAMPLES/sample.jsonl" --db "$DB"

step stats
$PREFPAIRS stats --db "$DB"

step rank --truth "$TRUTH"
bt=$($PREFPAIRS rank --db "$DB" --truth "$TRUTH")
printf '%s\n' "$bt"

step rank --method elo -x ann-06 -x ann-08 -x ann-10 -x ann-11 --truth "$TRUTH"
elo=$($PREFPAIRS rank --db "$DB" --method elo -x ann-06 -x ann-08 -x ann-10 -x ann-11 --truth "$TRUTH")
printf '%s\n' "$elo"

step checks --truth "$TRUTH"
checks=$($PREFPAIRS checks --db "$DB" --truth "$TRUTH")
printf '%s\n' "$checks"

for result in "$bt" "$elo"; do
  case "$result" in
    *"$EXPECT"*) ;;
    *) echo "demo FAILED: a ranking did not recover the true model order" >&2; exit 1 ;;
  esac
done
case "$checks" in
  *"$EXPECT_FLAGS"*) ;;
  *) echo "demo FAILED: the checks did not flag exactly the planted biased annotators" >&2; exit 1 ;;
esac
printf '\ndemo OK: both rankings recover the true model order and the checks flag the planted biases\n'
