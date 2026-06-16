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
  if command -v curl >/dev/null 2>&1; then
      setup_auth_curl
      while read -r line; do
        # Get everything after the last '/'
        filename="${line##*/}"

        # Strip everything after '?'
        stripped_query_params="${filename%%\?*}"

        curl -f -b "$cookiejar" -c "$cookiejar" -L --netrc-file "$netrc" -g -o "$OUTDIR/$stripped_query_params" -- $line && echo || exit_with_error "Command failed with error. Please retrieve the data manually."
      done;
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

        wget --load-cookies "$cookiejar" --save-cookies "$cookiejar" --output-document "$OUTDIR/$stripped_query_params" --keep-session-cookies -- $line && echo || exit_with_error "Command failed with error. Please retrieve the data manually."
      done;
  else
      exit_with_error "Error: Could not find a command-line downloader.  Please install curl or wget"
  fi
}

fetch_urls <<'EDSCEOF'
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_039855_2016030820160308_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_039855_2016030820160308_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_041404_2016030820160308_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_041404_2016030820160308_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_046644_2016030820160308_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_046644_2016030820160308_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_041404_2016030820160308_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_041404_2016030820160308_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_038456_2016030820160308_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_038456_2016030820160308_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_041991_2016030820160308_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_041991_2016030820160308_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_049095_2016030820160308_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_049095_2016030820160308_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_040932_2016030820160308_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_040932_2016030820160308_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_045612_2016030820160308_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_045612_2016030820160308_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_038915_2016030820160308_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_038915_2016030820160308_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_040483_2016030820160308_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_040483_2016030820160308_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_038456_2016030820160308_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_038456_2016030820160308_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_044874_2016030820160308_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_044874_2016030820160308_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_045612_2016030820160308_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_045612_2016030820160308_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_044874_2016030820160308_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_044874_2016030820160308_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_044874_2016030820160308_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_044874_2016030820160308_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_039855_2016030820160308_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_039855_2016030820160308_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_044165_2016030820160308_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_044165_2016030820160308_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_038002_2016030820160308_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_038002_2016030820160308_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_049095_2016030820160308_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_049095_2016030820160308_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_046644_2016030820160308_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_046644_2016030820160308_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_047404_2016030820160308_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_047404_2016030820160308_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_038002_2016030820160308_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_038002_2016030820160308_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_039434_2016030820160308_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_039434_2016030820160308_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_042643_2016030820160308_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_042643_2016030820160308_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_048492_2016030820160308_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_048492_2016030820160308_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_044165_2016030820160308_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_044165_2016030820160308_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_039855_2016030820160308_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_039855_2016030820160308_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_044165_2016030820160308_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_044165_2016030820160308_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_040932_2016030820160308_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_040932_2016030820160308_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_040483_2016030820160308_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_040483_2016030820160308_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_041991_2016030820160308_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_041991_2016030820160308_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_038002_2016030820160308_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_038002_2016030820160308_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_038456_2016030820160308_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_038456_2016030820160308_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_038915_2016030820160308_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_038915_2016030820160308_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_041991_2016030820160308_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_041991_2016030820160308_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_039434_2016030820160308_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_039434_2016030820160308_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_045612_2016030820160308_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_045612_2016030820160308_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_042643_2016030820160308_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_042643_2016030820160308_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_048492_2016030820160308_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_048492_2016030820160308_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_047404_2016030820160308_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_047404_2016030820160308_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_038915_2016030820160308_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_038915_2016030820160308_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_041404_2016030820160308_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_041404_2016030820160308_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_042643_2016030820160308_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_042643_2016030820160308_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_047404_2016030820160308_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_047404_2016030820160308_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_040932_2016030820160308_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_040932_2016030820160308_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_046644_2016030820160308_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_046644_2016030820160308_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_048492_2016030820160308_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_048492_2016030820160308_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_040483_2016030820160308_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_040483_2016030820160308_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_049095_2016030820160308_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_049095_2016030820160308_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_039434_2016030820160308_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_039434_2016030820160308_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_049990_2016030720160307_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_049990_2016030720160307_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_042782_2016030720160307_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_042782_2016030720160307_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_048713_2016030720160307_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_048713_2016030720160307_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_048713_2016030720160307_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_048713_2016030720160307_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_047175_2016030720160307_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_047175_2016030720160307_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_046578_2016030720160307_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_046578_2016030720160307_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_046578_2016030720160307_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_046578_2016030720160307_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_050613_2016030720160307_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_050613_2016030720160307_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_044948_2016030720160307_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_044948_2016030720160307_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_044074_2016030720160307_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_044074_2016030720160307_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_050613_2016030720160307_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_050613_2016030720160307_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_038009_2016030720160307_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_038009_2016030720160307_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_042782_2016030720160307_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_042782_2016030720160307_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_0378087_2016030720160307_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_0378087_2016030720160307_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_044948_2016030720160307_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_044948_2016030720160307_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_041057_2016030720160307_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_041057_2016030720160307_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_048044_2016030720160307_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_048044_2016030720160307_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_0378087_2016030720160307_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_0378087_2016030720160307_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_049990_2016030720160307_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_049990_2016030720160307_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_041927_2016030720160307_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_041927_2016030720160307_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_047175_2016030720160307_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_047175_2016030720160307_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_045866_2016030720160307_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_045866_2016030720160307_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_048713_2016030720160307_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_048713_2016030720160307_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_045866_2016030720160307_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_045866_2016030720160307_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_045866_2016030720160307_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_045866_2016030720160307_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_0378087_2016030720160307_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_0378087_2016030720160307_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_039209_2016030720160307_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_039209_2016030720160307_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_038009_2016030720160307_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_038009_2016030720160307_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_041057_2016030720160307_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_041057_2016030720160307_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_042782_2016030720160307_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_042782_2016030720160307_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_044074_2016030720160307_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_044074_2016030720160307_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_039209_2016030720160307_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_039209_2016030720160307_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_044948_2016030720160307_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_044948_2016030720160307_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_041927_2016030720160307_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_041927_2016030720160307_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_039209_2016030720160307_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_039209_2016030720160307_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_040075_2016030720160307_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_040075_2016030720160307_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_040075_2016030720160307_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_040075_2016030720160307_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_049387_2016030720160307_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_049387_2016030720160307_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_041927_2016030720160307_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_041927_2016030720160307_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_040075_2016030720160307_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_040075_2016030720160307_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_049387_2016030720160307_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_049387_2016030720160307_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_038009_2016030720160307_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_038009_2016030720160307_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_049990_2016030720160307_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_049990_2016030720160307_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_044074_2016030720160307_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_044074_2016030720160307_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_050613_2016030720160307_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_050613_2016030720160307_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_048044_2016030720160307_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_048044_2016030720160307_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_049387_2016030720160307_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_049387_2016030720160307_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_047175_2016030720160307_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_047175_2016030720160307_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_046578_2016030720160307_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_046578_2016030720160307_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_041057_2016030720160307_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_041057_2016030720160307_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_048044_2016030720160307_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_048044_2016030720160307_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_050443_2016030420160304_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_050443_2016030420160304_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_047142_2016030420160304_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_047142_2016030420160304_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_048534_2016030420160304_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_048534_2016030420160304_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_051215_2016030420160304_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_051215_2016030420160304_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_047931_2016030420160304_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_047931_2016030420160304_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_054146_2016030420160304_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_054146_2016030420160304_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_045395_2016030420160304_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_045395_2016030420160304_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_044946_2016030420160304_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_044946_2016030420160304_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_049074_2016030420160304_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_049074_2016030420160304_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_048534_2016030420160304_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_048534_2016030420160304_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_049833_2016030420160304_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_049833_2016030420160304_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_047142_2016030420160304_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_047142_2016030420160304_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_046693_2016030420160304_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_046693_2016030420160304_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_043591_2016030420160304_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_043591_2016030420160304_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_047142_2016030420160304_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_047142_2016030420160304_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_043591_2016030420160304_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_043591_2016030420160304_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_054146_2016030420160304_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_054146_2016030420160304_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_051732_2016030420160304_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_051732_2016030420160304_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_055158_2016030420160304_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_055158_2016030420160304_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_052387_2016030420160304_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_052387_2016030420160304_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_048534_2016030420160304_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_048534_2016030420160304_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_046693_2016030420160304_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_046693_2016030420160304_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_044284_2016030420160304_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_044284_2016030420160304_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_049074_2016030420160304_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_049074_2016030420160304_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_054746_2016030420160304_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_054746_2016030420160304_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_052387_2016030420160304_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_052387_2016030420160304_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_044946_2016030420160304_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_044946_2016030420160304_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_044946_2016030420160304_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_044946_2016030420160304_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_050443_2016030420160304_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_050443_2016030420160304_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_051732_2016030420160304_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_051732_2016030420160304_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_049074_2016030420160304_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_049074_2016030420160304_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_053344_2016030420160304_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_053344_2016030420160304_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_045395_2016030420160304_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_045395_2016030420160304_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_046041_2016030420160304_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_046041_2016030420160304_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_053344_2016030420160304_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_053344_2016030420160304_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_046041_2016030420160304_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_046041_2016030420160304_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_055158_2016030420160304_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_055158_2016030420160304_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_049833_2016030420160304_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_049833_2016030420160304_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_044284_2016030420160304_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_044284_2016030420160304_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_054746_2016030420160304_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_054746_2016030420160304_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_050443_2016030420160304_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_050443_2016030420160304_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_047931_2016030420160304_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_047931_2016030420160304_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_053344_2016030420160304_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_053344_2016030420160304_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_044284_2016030420160304_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_044284_2016030420160304_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_045395_2016030420160304_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_045395_2016030420160304_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_054746_2016030420160304_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_054746_2016030420160304_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_051215_2016030420160304_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_051215_2016030420160304_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_051732_2016030420160304_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_051732_2016030420160304_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_052387_2016030420160304_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_052387_2016030420160304_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_054146_2016030420160304_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_054146_2016030420160304_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_043591_2016030420160304_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_043591_2016030420160304_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_047931_2016030420160304_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_047931_2016030420160304_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_046693_2016030420160304_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_046693_2016030420160304_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_046041_2016030420160304_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_046041_2016030420160304_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_049833_2016030420160304_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_049833_2016030420160304_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_051215_2016030420160304_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_051215_2016030420160304_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_055158_2016030420160304_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_055158_2016030420160304_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_058029_2016030220160302_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_058029_2016030220160302_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_060055_2016030220160302_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_060055_2016030220160302_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_059019_2016030220160302_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_059019_2016030220160302_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_065027_2016030220160302_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_065027_2016030220160302_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_056989_2016030220160302_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_056989_2016030220160302_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_061654_2016030220160302_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_061654_2016030220160302_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_062278_2016030220160302_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_062278_2016030220160302_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_059019_2016030220160302_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_059019_2016030220160302_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_065695_2016030220160302_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_065695_2016030220160302_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_056408_2016030220160302_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_056408_2016030220160302_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_064622_2016030220160302_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_064622_2016030220160302_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_061248_2016030220160302_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_061248_2016030220160302_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_063699_2016030220160302_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_063699_2016030220160302_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_058029_2016030220160302_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_058029_2016030220160302_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_061654_2016030220160302_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_061654_2016030220160302_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_062278_2016030220160302_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_062278_2016030220160302_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_066354_2016030220160302_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_066354_2016030220160302_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_059646_2016030220160302_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_059646_2016030220160302_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_059019_2016030220160302_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_059019_2016030220160302_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_063295_2016030220160302_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_063295_2016030220160302_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_056408_2016030220160302_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_056408_2016030220160302_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_060055_2016030220160302_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_060055_2016030220160302_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_058600_2016030220160302_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_058600_2016030220160302_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_065027_2016030220160302_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_065027_2016030220160302_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_060655_2016030220160302_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_060655_2016030220160302_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_065027_2016030220160302_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_065027_2016030220160302_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_065695_2016030220160302_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_065695_2016030220160302_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_056989_2016030220160302_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_056989_2016030220160302_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_061248_2016030220160302_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_061248_2016030220160302_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_056408_2016030220160302_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_056408_2016030220160302_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_062278_2016030220160302_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_062278_2016030220160302_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_062682_2016030220160302_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_062682_2016030220160302_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_064622_2016030220160302_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_064622_2016030220160302_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_055987_2016030220160302_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_055987_2016030220160302_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_058029_2016030220160302_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_058029_2016030220160302_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_061248_2016030220160302_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_061248_2016030220160302_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_060655_2016030220160302_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_060655_2016030220160302_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_055987_2016030220160302_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_055987_2016030220160302_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_066354_2016030220160302_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_066354_2016030220160302_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_065695_2016030220160302_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_065695_2016030220160302_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_057582_2016030220160302_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_057582_2016030220160302_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_058600_2016030220160302_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_058600_2016030220160302_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_057582_2016030220160302_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_057582_2016030220160302_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_059646_2016030220160302_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_059646_2016030220160302_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_058600_2016030220160302_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_058600_2016030220160302_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_056989_2016030220160302_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_056989_2016030220160302_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_066354_2016030220160302_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_066354_2016030220160302_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_059646_2016030220160302_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_059646_2016030220160302_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_062682_2016030220160302_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_062682_2016030220160302_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_055488_2016030220160302_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_055488_2016030220160302_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_055488_2016030220160302_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_055488_2016030220160302_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_061654_2016030220160302_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_061654_2016030220160302_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_063295_2016030220160302_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_063295_2016030220160302_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_057582_2016030220160302_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_057582_2016030220160302_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_063295_2016030220160302_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_063295_2016030220160302_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_055987_2016030220160302_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_055987_2016030220160302_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_063699_2016030220160302_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_063699_2016030220160302_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_060655_2016030220160302_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_060655_2016030220160302_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_062682_2016030220160302_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_062682_2016030220160302_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_055488_2016030220160302_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_055488_2016030220160302_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_063699_2016030220160302_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_063699_2016030220160302_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_060055_2016030220160302_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_060055_2016030220160302_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_064622_2016030220160302_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_064622_2016030220160302_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_048056_2016022020160220_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_048056_2016022020160220_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_038619_2016022020160220_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_038619_2016022020160220_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_038024_2016022020160220_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_038024_2016022020160220_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_040422_2016022020160220_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_040422_2016022020160220_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_044415_2016022020160220_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_044415_2016022020160220_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_047064_2016022020160220_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_047064_2016022020160220_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_038024_2016022020160220_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_038024_2016022020160220_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_044415_2016022020160220_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_044415_2016022020160220_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_048720_2016022020160220_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_048720_2016022020160220_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_045956_2016022020160220_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_045956_2016022020160220_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_046437_2016022020160220_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_046437_2016022020160220_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_045956_2016022020160220_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_045956_2016022020160220_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_039527_2016022020160220_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_039527_2016022020160220_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_042984_2016022020160220_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_042984_2016022020160220_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_042984_2016022020160220_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_042984_2016022020160220_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_040422_2016022020160220_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_040422_2016022020160220_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_042155_2016022020160220_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_042155_2016022020160220_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_041303_2016022020160220_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_041303_2016022020160220_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_043873_2016022020160220_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_043873_2016022020160220_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_042155_2016022020160220_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_042155_2016022020160220_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_039527_2016022020160220_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_039527_2016022020160220_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_050005_2016022020160220_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_050005_2016022020160220_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_050005_2016022020160220_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_050005_2016022020160220_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_039527_2016022020160220_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_039527_2016022020160220_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_038619_2016022020160220_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_038619_2016022020160220_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_048056_2016022020160220_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_048056_2016022020160220_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_040422_2016022020160220_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_040422_2016022020160220_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_041303_2016022020160220_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_041303_2016022020160220_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_042155_2016022020160220_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_042155_2016022020160220_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_045956_2016022020160220_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_045956_2016022020160220_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_038619_2016022020160220_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_038619_2016022020160220_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_043873_2016022020160220_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_043873_2016022020160220_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_045137_2016022020160220_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_045137_2016022020160220_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_043873_2016022020160220_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_043873_2016022020160220_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_046437_2016022020160220_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_046437_2016022020160220_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_038024_2016022020160220_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_038024_2016022020160220_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_047064_2016022020160220_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_047064_2016022020160220_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_050005_2016022020160220_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_050005_2016022020160220_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_049158_2016022020160220_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_049158_2016022020160220_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_048056_2016022020160220_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_048056_2016022020160220_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_046437_2016022020160220_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_046437_2016022020160220_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_047064_2016022020160220_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_047064_2016022020160220_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_044415_2016022020160220_l2a_metrics_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_044415_2016022020160220_l2a_metrics_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_049158_2016022020160220_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_049158_2016022020160220_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_048720_2016022020160220_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_048720_2016022020160220_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_049158_2016022020160220_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_049158_2016022020160220_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_041303_2016022020160220_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_041303_2016022020160220_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_045137_2016022020160220_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_045137_2016022020160220_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_042984_2016022020160220_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_042984_2016022020160220_l2b_covz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_048720_2016022020160220_l2b_paiz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_048720_2016022020160220_l2b_paiz_e04326_v0100.csv.sha256
https://data.ornldaac.earthdata.nasa.gov/protected/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_045137_2016022020160220_l2b_covz_e04326_v0100.csv
https://data.ornldaac.earthdata.nasa.gov/public/afrisar/AfriSAR_LVIS_Footprint_Cover/data/lvis2_045137_2016022020160220_l2b_covz_e04326_v0100.csv.sha256
EDSCEOF
