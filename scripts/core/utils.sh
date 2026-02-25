get_options() {

if [[ "$1" == "list" || -z "$1" ]]; then
    echo "--- Available Options ---"
    # 1. Finds the case (e.g., 00)
    # 2. Skips the first comment line (# ====)
    # 3. Prints the second comment line
    awk '/^[[:space:]]*[0-9a-zA-Z._-]+\)/ && !/case/ {
        case_val = $0;
        getline; # skip the # === line
        getline; # get the description line
        print case_val " " $0;
    }' "$0" | sed 's/)//g'
    exit 0
fi
}