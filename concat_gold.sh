#!/bin/bash

# Hardcoded paths
GOLD_DIR="gold_standard"
OUTPUT_FILE="$GOLD_DIR/!concat_gold.txt"

# Ensure the gold standard directory exists
if [ ! -d "$GOLD_DIR" ]; then
    echo "Error: Directory '$GOLD_DIR' not found!"
    exit 1
fi

# Clear the output file before writing
> "$OUTPUT_FILE"

# Iterate through sorted files and concatenate
first_file=true
find "$GOLD_DIR" -type f -not -name "README.md" | sort | while read -r file; do
    if [ "$first_file" = true ]; then
        first_file=false
    else
        echo >> "$OUTPUT_FILE"  # Add exactly one newline between files
    fi
    awk 'NF {p=1} p' "$file" >> "$OUTPUT_FILE"  # Remove leading/trailing empty lines
done

# Check if output file is empty
if [ ! -s "$OUTPUT_FILE" ]; then
    echo "Warning: No valid files found to concatenate!"
else
    echo "Concatenation complete. Output saved to $OUTPUT_FILE."
fi