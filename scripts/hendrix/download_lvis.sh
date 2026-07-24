#!/bin/bash

GREP_OPTIONS=''

cookiejar=$(mktemp cookies.XXXXXXXXXX)
netrc=$(mktemp netrc.XXXXXXXXXX)
chmod 0600 "$cookiejar" "$netrc"
function finish {
  rm -rf "$cookiejar" "$netrc"
}

trap finish EXIT
WGETRC="$wgetrc"

prompt_credentials() {
    echo "Enter your Earthdata Login or other provider supplied credentials"
    read -p "Username (huizh): " username
    username=${username:-huizh}
    read -s -p "Password: " password
    echo "machine urs.earthdata.nasa.gov login $username password $password" >> $netrc
    echo
}

exit_with_error() {
    echo
    echo "Unable to Retrieve Data"
    echo
    echo $1
    echo
    echo "https://data.nsidc.earthdatacloud.nasa.gov/nsidc-cumulus-prod-protected/LVIS/AFLVIS2/1/2016/03/08/LVIS2_Gabon2016_0308_R1808_049095.TXT"
    echo
    exit 1
}

OUTDIR="${OUTDIR:-.}"
mkdir -p "$OUTDIR"

# Directory holding the URL list files (one URL per line; blank lines / '#' comments ignored).
URLDIR="${URLDIR:-$(cd "$(dirname "$0")" && pwd)/lvis_urls}"

# Dataset registry. One entry per line: "key|dest-subdir|url-list-file".
# To add a dataset, drop a <name>.txt file in $URLDIR and add a line here.
BATCHES=(
    "afrisar2016|AFLVIS2_AfriSAR2016|afrisar2016.txt"
    "gabon2023|LVISC2_Gabon2023|gabon2023.txt"
    "gedi2019|LVISC2_GEDI2019|gedi2019.txt"
    "gedi2021|LVISC2_GEDI2021|gedi2021.txt"
)

# Which dataset(s) to download: a key from BATCHES above, or "all".
BATCH="${BATCH:-all}"
if [ "$BATCH" != "all" ]; then
    valid=0
    for entry in "${BATCHES[@]}"; do
        [ "${entry%%|*}" = "$BATCH" ] && valid=1
    done
    if [ "$valid" -ne 1 ]; then
        keys=$(printf '%s\n' "${BATCHES[@]}" | cut -d'|' -f1 | paste -sd'|' -)
        echo "Invalid BATCH='$BATCH' (expected: $keys | all)"; exit 1
    fi
fi

prompt_credentials
  detect_app_approval() {
    approved=`curl -s -b "$cookiejar" -c "$cookiejar" -L --max-redirs 5 --netrc-file "$netrc" https://data.nsidc.earthdatacloud.nasa.gov/nsidc-cumulus-prod-protected/LVIS/AFLVIS2/1/2016/03/08/LVIS2_Gabon2016_0308_R1808_049095.TXT -w '\n%{http_code}' | tail  -1`
    if [ "$approved" -ne "200" ] && [ "$approved" -ne "301" ] && [ "$approved" -ne "302" ]; then
        # User didn't approve the app. Direct users to approve the app in URS
        exit_with_error "Please ensure that you have authorized the remote application by visiting the link below "
    fi
}

setup_auth_curl() {
    # Firstly, check if it require URS authentication
    status=$(curl -s -z "$(date)" -w '\n%{http_code}' https://data.nsidc.earthdatacloud.nasa.gov/nsidc-cumulus-prod-protected/LVIS/AFLVIS2/1/2016/03/08/LVIS2_Gabon2016_0308_R1808_049095.TXT | tail -1)
    if [[ "$status" -ne "200" && "$status" -ne "304" ]]; then
        # URS authentication is required. Now further check if the application/remote service is approved.
        detect_app_approval
    fi
}

setup_auth_wget() {
    # The safest way to auth via curl is netrc. Note: there's no checking or feedback
    # if login is unsuccessful
    touch ~/.netrc
    chmod 0600 ~/.netrc
    credentials=$(grep 'machine urs.earthdata.nasa.gov' ~/.netrc)
    if [ -z "$credentials" ]; then
        cat "$netrc" >> ~/.netrc
    fi
}

fetch_urls() {
  # $1: destination directory for this batch (created if missing)
  # $2: path to a text file of URLs, one per line (blank lines and '#' comments ignored)
  dest="${1:-$OUTDIR}"
  urlfile="$2"
  mkdir -p "$dest"
  if [ ! -f "$urlfile" ]; then
      exit_with_error "URL list file not found: $urlfile"
  fi
  if command -v curl >/dev/null 2>&1; then
      setup_auth_curl
      while read -r line; do
        # Get everything after the last '/'
        filename="${line##*/}"

        # Strip everything after '?'
        stripped_query_params="${filename%%\?*}"

        curl -f -b "$cookiejar" -c "$cookiejar" -L --netrc-file "$netrc" -g -o "$dest/$stripped_query_params" -- $line && echo || exit_with_error "Command failed with error. Please retrieve the data manually."
      done < <(grep -vE '^[[:space:]]*(#|$)' "$urlfile");
  elif command -v wget >/dev/null 2>&1; then
      # We can't use wget to poke provider server to get info whether or not URS was integrated without download at least one of the files.
      echo
      echo "WARNING: Can't find curl, use wget instead."
      echo "WARNING: Script may not correctly identify Earthdata Login integrations."
      echo
      setup_auth_wget
      while read -r line; do
        # Get everything after the last '/'
        filename="${line##*/}"

        # Strip everything after '?'
        stripped_query_params="${filename%%\?*}"

        wget --load-cookies "$cookiejar" --save-cookies "$cookiejar" --output-document "$dest/$stripped_query_params" --keep-session-cookies -- $line && echo || exit_with_error "Command failed with error. Please retrieve the data manually."
      done < <(grep -vE '^[[:space:]]*(#|$)' "$urlfile");
  else
      exit_with_error "Error: Could not find a command-line downloader.  Please install curl or wget"
  fi
}

# ---- Download the selected dataset(s) ----
for entry in "${BATCHES[@]}"; do
    key="${entry%%|*}"
    rest="${entry#*|}"
    subdir="${rest%%|*}"
    urlfile="${rest##*|}"

    if [ "$BATCH" = "all" ] || [ "$BATCH" = "$key" ]; then
        echo "==> Downloading '$key' into $OUTDIR/$subdir"
        fetch_urls "$OUTDIR/$subdir" "$URLDIR/$urlfile"
    fi
done
