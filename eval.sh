#!/bin/bash
#SBATCH --time=1:00:00 --output=eval.out --error=eval.err
#SBATCH --mem=5GB
#SBATCH -c 5

AMR_TEST_DIR=amr_data/amr_annotation_3.0/data/amrs/split/test
if [ ! -e $AMR_TEST_DIR ]
then
    echo "Error: $AMR_TEST_DIR does not exist"
    exit 1
fi

PRED_FILE="gold_standard/concat_pred.txt"

# Get SLURM_TMPDIR, or else TMPDIR, or else local directory
SCRIPT_TMPDIR=${SLURM_TMPDIR:-${TMPDIR:-.}}
echo "Using $SCRIPT_TMPDIR" to write temporary files

GOLD_FILE="$SCRIPT_TMPDIR/!concat_gold.txt"
cat $(ls -d $AMR_TEST_DIR/* | sort) > $GOLD_FILE
echo "Wrote concatenated AMR 3.0 data to $GOLD_FILE..."
# exit 0

# Ensure both files exist
if [ ! -f "$GOLD_FILE" ]; then
    echo "Error: Gold standard file '$GOLD_FILE' not found!"
    exit 1
fi

if [ ! -f "$PRED_FILE" ]; then
    echo "Error: Prediction file '$PRED_FILE' not found!"
    exit 1
fi

# Run metrics
echo "Running evaluation metrics..."

echo "SMATCH:"
smatch.py -f "$PRED_FILE" "$GOLD_FILE"

echo "SemBLEU:"
python3 sembleu/src/eval.py "$PRED_FILE" "$GOLD_FILE"

echo "Fine-Grained AMR-Evaluation:"
sed 's/:[a-zA-Z0-9-]*/:label/g' "$PRED_FILE" > 1.tmp
sed 's/:[a-zA-Z0-9-]*/:label/g' "$GOLD_FILE" > 2.tmp
out=`smatch.py --pr -f 1.tmp 2.tmp`
pr=`echo $out | cut -d' ' -f2`
rc=`echo $out | cut -d' ' -f4`
fs=`echo $out | cut -d' ' -f6`
echo 'Unlabeled -> P: '$pr', R: '$rc', F: '$fs 

cat "$PRED_FILE" | perl -ne 's/(\/ [a-zA-Z0-9\-][a-zA-Z0-9\-]*)-[0-9][0-9]*/\1-01/g; print;' > 1.tmp
cat "$GOLD_FILE" | perl -ne 's/(\/ [a-zA-Z0-9\-][a-zA-Z0-9\-]*)-[0-9][0-9]*/\1-01/g; print;' > 2.tmp
out=`smatch.py --pr -f 1.tmp 2.tmp`
pr=`echo $out | cut -d' ' -f2`
rc=`echo $out | cut -d' ' -f4`
fs=`echo $out | cut -d' ' -f6`
echo 'No WSD -> -> P: '$pr', R: '$rc', F: '$fs

cat "$PRED_FILE" | perl -ne 's/^#.*\n//g; print;' | tr '\t' ' ' | tr -s ' ' > 1.tmp
cat "$GOLD_FILE" | perl -ne 's/^#.*\n//g; print;' | tr '\t' ' ' | tr -s ' ' > 2.tmp
python amr-evaluation/scores.py "1.tmp" "2.tmp"

rm 1.tmp
rm 2.tmp

echo "Evaluation complete."
