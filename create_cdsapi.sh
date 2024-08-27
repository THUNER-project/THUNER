#!/bin/bash

cat <<EOF > ~/.cdsapirc
url: https://cds.climate.copernicus.eu/api/v2
key: $CDSAPI_KEY
EOF