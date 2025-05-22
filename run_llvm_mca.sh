#!/bin/bash
# Usage: ./run_llvm_mca.sh [input_json] [output_json]

# Default values
DEFAULT_INPUT_JSON="./random_generate/asm.json"
DEFAULT_OUTPUT_JSON="./random_generate/mca.json"

# Check parameters and set input/output files
if [ $# -eq 0 ]; then
    INPUT_JSON="$DEFAULT_INPUT_JSON"
    OUTPUT_JSON="$DEFAULT_OUTPUT_JSON"
elif [ $# -eq 1 ]; then
    INPUT_JSON="$1"
    OUTPUT_JSON="$DEFAULT_OUTPUT_JSON"
elif [ $# -eq 2 ]; then
    INPUT_JSON="$1"
    OUTPUT_JSON="$2"
else
    echo "Error: Too many parameters"
    echo "Usage: $0 [input_json] [output_json]"
    exit 1
fi

TEMP_DIR=$(mktemp -d)
TEMP_ASM="$TEMP_DIR/temp.S"

# Ensure output directory exists
OUTPUT_DIR=$(dirname "$OUTPUT_JSON")
mkdir -p "$OUTPUT_DIR"

# Clean up temporary files on exit
trap 'rm -rf "$TEMP_DIR"' EXIT

# Check if input file exists
if [ ! -f "$INPUT_JSON" ]; then
    echo "Error: Input file '$INPUT_JSON' does not exist"
    exit 1
fi

# Check if llvm-mca is executable
if ! command -v llvm-mca &> /dev/null; then
    echo "Error: '/mnt/d/riscv/bin/llvm-mca' does not exist or is not executable"
    exit 1
fi

# Initialize output JSON
echo "[" > "$OUTPUT_JSON"
first_entry=true

# Parse JSON input using jq (must be installed)
if ! command -v jq &> /dev/null; then
    echo "Error: 'jq' is required but not installed. Please install jq first."
    exit 1
fi

# Get total number of entries
total_entries=$(jq '. | length' "$INPUT_JSON")
echo "Found $total_entries entries to process..."

# Process each entry
for i in $(seq 0 $(($total_entries - 1))); do
    # Extract asm content
    asm_content=$(jq -r ".[$i].asm" "$INPUT_JSON")

    # Write asm to temporary file
    echo -e "$asm_content" > "$TEMP_ASM"

    # Run llvm-mca
    mca_output=$(llvm-mca -mcpu=xiangshan-nanhu -iterations=1000 --instruction-info=0 -resource-pressure=0 "$TEMP_ASM" 2>&1)

    # Check if command was successful
    if [ $? -ne 0 ]; then
        mca_output="Error running llvm-mca: $mca_output"
    fi

    # Add comma separator if not the first entry
    if [ "$first_entry" = false ]; then
        echo "," >> "$OUTPUT_JSON"
    fi
    first_entry=false

    # Write result to output JSON
    jq -n --arg asm "$asm_content" --arg mca "$mca_output" \
       '{"asm": $asm, "mca_result": $mca}' >> "$OUTPUT_JSON"

#    echo "Processed entry $((i+1))/$total_entries"
done

# Close the JSON array
echo "]" >> "$OUTPUT_JSON"

echo "Processing complete. Results saved to $OUTPUT_JSON"
exit 0