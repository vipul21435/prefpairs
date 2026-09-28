#!/bin/sh
# End-to-end PrefPairs demo on the bundled sample: a seed-0 simulation with
# known ground truth (6 models, 12 annotators, 4 of them planted as defective).
# It imports the JSONL, summarises it, ranks the models with Bradley-Terry and
# with Elo (leaving the planted annotators out), runs the annotator checks and
# the combined audit, and fails unless both rankings recover the true model
# order, the checks flag exactly the planted position-biased and length-biased
# annotators, and the audit flags exactly the four planted annotators (and
# exits 1 under --strict). It then exports DPO, KTO and reward-model data from
# the unflagged annotators and fails unless a second DPO export is
# byte-identical to the first.
#
#   PREFPAIRS  command that runs the CLI (default: prefpairs)
#   DEMO_DB    database to (re)create      (default: .prefpairs/demo.db)
#   EXAMPLES   directory of the sample     (default: examples)
#   EXPORT_DIR where exports are written    (default: .prefpairs/demo-export)
set -eu

PREFPAIRS="${PREFPAIRS:-prefpairs}"
DB="${DEMO_DB:-.prefpairs/demo.db}"
EXAMPLES="${EXAMPLES:-examples}"
EXPORT_DIR="${EXPORT_DIR:-.prefpairs/demo-export}"
TRUTH="$EXAMPLES/sample-truth.json"
EXPECT="Kendall tau against the true order: 1.000"
EXPECT_FLAGS="flagged: ann-10 (position), ann-11 (length)"
EXPECT_AUDIT="flagged: ann-06 (gold+reversed), ann-08 (intransitive+spammer), ann-10 (position), ann-11 (length)"

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

step audit --truth "$TRUTH"
audit=$($PREFPAIRS audit --db "$DB" --truth "$TRUTH")
printf '%s\n' "$audit"

step audit --strict
if $PREFPAIRS audit --db "$DB" --strict >/dev/null; then
  echo "demo FAILED: audit --strict exited 0 although annotators are flagged" >&2; exit 1
fi
echo "exit code 1 (annotators flagged)"

rm -rf "$EXPORT_DIR"
for format in dpo kto rm; do
  step export --format "$format" --out "$EXPORT_DIR"
  $PREFPAIRS export --db "$DB" --format "$format" --out "$EXPORT_DIR"
done
$PREFPAIRS export --db "$DB" --format dpo --out "$EXPORT_DIR/rerun" >/dev/null
for name in train.jsonl validation.jsonl test.jsonl card.json card.md; do
  if ! cmp -s "$EXPORT_DIR/dpo/$name" "$EXPORT_DIR/rerun/dpo/$name"; then
    echo "demo FAILED: a second DPO export changed $name" >&2; exit 1
  fi
done
echo "rerun: every DPO file and card is byte-identical"

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
case "$audit" in
  *"$EXPECT_AUDIT"*) ;;
  *) echo "demo FAILED: the audit did not flag exactly the four planted annotators" >&2; exit 1 ;;
esac
printf '\ndemo OK: both rankings recover the true model order, the audit flags exactly the planted annotators and the export is reproducible\n'
