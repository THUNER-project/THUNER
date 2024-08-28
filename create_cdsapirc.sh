#!/bin/bash

cat <<EOF > ~/.cdsapirc
url: https://cds-beta.climate.copernicus.eu/api
key: $CDSAPI_KEY
EOF