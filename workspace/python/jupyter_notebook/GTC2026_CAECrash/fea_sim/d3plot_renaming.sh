#!/bin/sh

# Input folders
BASE_DIR="./dataset"

for RUN_DIR in "$BASE_DIR"/run*/; do
    # Check if the directory exists to avoid errors if no run* folders are found
    [ -d "$RUN_DIR" ] || continue

    echo "Processing files in: $RUN_DIR"

    # Loop all matching input files
    for FILE in "${RUN_DIR}"Bumper_Beam_AP_meshed.d3plot*; do

        # Skip if no files match the pattern
        [ -e "$FILE" ] || continue

        FILENAME=$(basename "$FILE")

        # Extract suffix after ".d3plot"
        # Example: Bumper_Beam_AP_meshed.d3plot01 -> 01
        SUFFIX=$(printf "%s" "$FILENAME" | sed 's/.*\.d3plot//')

        # Determine new filename
        if [ "$SUFFIX" = "$FILENAME" ]; then
            continue     # Safety skip: pattern didn't match as expected
        elif [ "$SUFFIX" = "" ]; then
            NEW_NAME="d3plot"
        else
            NEW_NAME="d3plot$SUFFIX"
        fi

        # Perform the in-place rename
        printf "\rRenaming: %-40s → %-15s" "$FILENAME" "$NEW_NAME"
        mv "$FILE" "${RUN_DIR}${NEW_NAME}"
    done
    printf "\n"
done

echo "In-place renaming complete."