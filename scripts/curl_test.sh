endpoint="https://s3.twcc.ai:443"
access_key="5QL09M2O1Y8E4GTOFC9Z"
secret_key="9mXMT1kJAYAzOGZusIc5CT856cc3O22FqaYZpeTN"

# endpoint="https://200-9090.aifs.ym.wise-paas.com"
# access_key="mDUtLL4OCkM4CjieIxlxozvTkpp8r06L"
# secret_key="Q3CwL8dYmX5i1jSbnSQrU29bHtRPr0BT"

bucket="test-bucket-clara"
output_s3_folder="Inference_Source/2022-01-07_10:15:00.000"


# api_url="https://11-33260.api-aifs-ai001.education.wise-paas.com"
api_url="http://localhost:1234"

# organs="[\"liver\", \"pancreas\"]"
# organs="[\"liver\", \"pancreas\", \"spleen\"]"
# curl -X POST $api_url/stop
# curl -X POST $api_url/model
# curl -X POST $api_url/predict --header "Content-Type: application/json" -k --data "{\"organs\" : $organs, \"endpoint\" : \"$endpoint\", \"access_key\" : \"$access_key\", \"secret_key\" : \"$secret_key\", \"bucket\" : \"$bucket\", \"output_s3_folder\": \"$output_s3_folder\", \"asset_group\": [{\"category_name\": \"001\", \"files\": [\"advantech_aifs_aiaa/001/I0182471.dcm\", \"advantech_aifs_aiaa/001/I0182472.dcm\", \"advantech_aifs_aiaa/001/I0182473.dcm\",\"advantech_aifs_aiaa/001/I0182474.dcm\", \"advantech_aifs_aiaa/001/I0182475.dcm\", \"advantech_aifs_aiaa/001/I0182476.dcm\", \"advantech_aifs_aiaa/001/I0182477.dcm\", \"advantech_aifs_aiaa/001/I0182478.dcm\", \"advantech_aifs_aiaa/001/I0182479.dcm\", \"advantech_aifs_aiaa/001/I0182480.dcm\", \"advantech_aifs_aiaa/001/I0182481.dcm\", \"advantech_aifs_aiaa/001/I0182482.dcm\", \"advantech_aifs_aiaa/001/I0182483.dcm\", \"advantech_aifs_aiaa/001/I0182484.dcm\", \"advantech_aifs_aiaa/001/I0182485.dcm\", \"advantech_aifs_aiaa/001/I0182486.dcm\", \"advantech_aifs_aiaa/001/I0182487.dcm\", \"advantech_aifs_aiaa/001/I0182488.dcm\", \"advantech_aifs_aiaa/001/I0182489.dcm\", \"advantech_aifs_aiaa/001/I0182490.dcm\", \"advantech_aifs_aiaa/001/I0182491.dcm\", \"advantech_aifs_aiaa/001/I0182492.dcm\", \"advantech_aifs_aiaa/001/I0182493.dcm\", \"advantech_aifs_aiaa/001/I0182494.dcm\", \"advantech_aifs_aiaa/001/I0182495.dcm\", \"advantech_aifs_aiaa/001/I0182496.dcm\", \"advantech_aifs_aiaa/001/I0182497.dcm\", \"advantech_aifs_aiaa/001/I0182498.dcm\", \"advantech_aifs_aiaa/001/I0182499.dcm\", \"advantech_aifs_aiaa/001/I0182500.dcm\", \"advantech_aifs_aiaa/001/I0182501.dcm\", \"advantech_aifs_aiaa/001/I0182502.dcm\", \"advantech_aifs_aiaa/001/I0182503.dcm\", \"advantech_aifs_aiaa/001/I0182504.dcm\", \"advantech_aifs_aiaa/001/I0182505.dcm\", \"advantech_aifs_aiaa/001/I0182506.dcm\", \"advantech_aifs_aiaa/001/I0182507.dcm\", \"advantech_aifs_aiaa/001/I0182508.dcm\"]}, {\"category_name\": \"002\", \"files\": [\"advantech_aifs_aiaa/002/I0182862.dcm\", \"advantech_aifs_aiaa/002/I0182863.dcm\", \"advantech_aifs_aiaa/002/I0182864.dcm\", \"advantech_aifs_aiaa/002/I0182865.dcm\", \"advantech_aifs_aiaa/002/I0182866.dcm\", \"advantech_aifs_aiaa/002/I0182867.dcm\", \"advantech_aifs_aiaa/002/I0182868.dcm\", \"advantech_aifs_aiaa/002/I0182869.dcm\", \"advantech_aifs_aiaa/002/I0182870.dcm\", \"advantech_aifs_aiaa/002/I0182871.dcm\", \"advantech_aifs_aiaa/002/I0182872.dcm\", \"advantech_aifs_aiaa/002/I0182873.dcm\", \"advantech_aifs_aiaa/002/I0182874.dcm\", \"advantech_aifs_aiaa/002/I0182875.dcm\", \"advantech_aifs_aiaa/002/I0182876.dcm\", \"advantech_aifs_aiaa/002/I0182877.dcm\", \"advantech_aifs_aiaa/002/I0182878.dcm\", \"advantech_aifs_aiaa/002/I0182879.dcm\", \"advantech_aifs_aiaa/002/I0182880.dcm\", \"advantech_aifs_aiaa/002/I0182881.dcm\", \"advantech_aifs_aiaa/002/I0182882.dcm\", \"advantech_aifs_aiaa/002/I0182883.dcm\", \"advantech_aifs_aiaa/002/I0182884.dcm\", \"advantech_aifs_aiaa/002/I0182885.dcm\", \"advantech_aifs_aiaa/002/I0182886.dcm\", \"advantech_aifs_aiaa/002/I0182887.dcm\", \"advantech_aifs_aiaa/002/I0182888.dcm\", \"advantech_aifs_aiaa/002/I0182889.dcm\", \"advantech_aifs_aiaa/002/I0182890.dcm\", \"advantech_aifs_aiaa/002/I0182891.dcm\", \"advantech_aifs_aiaa/002/I0182892.dcm\", \"advantech_aifs_aiaa/002/I0182893.dcm\", \"advantech_aifs_aiaa/002/I0182894.dcm\", \"advantech_aifs_aiaa/002/I0182895.dcm\", \"advantech_aifs_aiaa/002/I0182896.dcm\", \"advantech_aifs_aiaa/002/I0182897.dcm\", \"advantech_aifs_aiaa/002/I0182898.dcm\", \"advantech_aifs_aiaa/002/I0182899.dcm\", \"advantech_aifs_aiaa/002/I0182900.dcm\", \"advantech_aifs_aiaa/002/I0182901.dcm\", \"advantech_aifs_aiaa/002/I0182902.dcm\", \"advantech_aifs_aiaa/002/I0182903.dcm\"]}, {\"category_name\": \"003\", \"files\": [\"advantech_aifs_aiaa/003/I0183254.dcm\", \"advantech_aifs_aiaa/003/I0183255.dcm\", \"advantech_aifs_aiaa/003/I0183256.dcm\", \"advantech_aifs_aiaa/003/I0183257.dcm\", \"advantech_aifs_aiaa/003/I0183258.dcm\", \"advantech_aifs_aiaa/003/I0183259.dcm\", \"advantech_aifs_aiaa/003/I0183260.dcm\", \"advantech_aifs_aiaa/003/I0183261.dcm\", \"advantech_aifs_aiaa/003/I0183262.dcm\", \"advantech_aifs_aiaa/003/I0183263.dcm\", \"advantech_aifs_aiaa/003/I0183264.dcm\", \"advantech_aifs_aiaa/003/I0183265.dcm\", \"advantech_aifs_aiaa/003/I0183266.dcm\", \"advantech_aifs_aiaa/003/I0183267.dcm\", \"advantech_aifs_aiaa/003/I0183268.dcm\", \"advantech_aifs_aiaa/003/I0183269.dcm\", \"advantech_aifs_aiaa/003/I0183270.dcm\", \"advantech_aifs_aiaa/003/I0183271.dcm\", \"advantech_aifs_aiaa/003/I0183272.dcm\", \"advantech_aifs_aiaa/003/I0183273.dcm\", \"advantech_aifs_aiaa/003/I0183274.dcm\", \"advantech_aifs_aiaa/003/I0183275.dcm\", \"advantech_aifs_aiaa/003/I0183276.dcm\", \"advantech_aifs_aiaa/003/I0183277.dcm\", \"advantech_aifs_aiaa/003/I0183278.dcm\", \"advantech_aifs_aiaa/003/I0183279.dcm\", \"advantech_aifs_aiaa/003/I0183280.dcm\", \"advantech_aifs_aiaa/003/I0183281.dcm\", \"advantech_aifs_aiaa/003/I0183282.dcm\", \"advantech_aifs_aiaa/003/I0183283.dcm\", \"advantech_aifs_aiaa/003/I0183284.dcm\", \"advantech_aifs_aiaa/003/I0183285.dcm\", \"advantech_aifs_aiaa/003/I0183286.dcm\", \"advantech_aifs_aiaa/003/I0183287.dcm\", \"advantech_aifs_aiaa/003/I0183288.dcm\", \"advantech_aifs_aiaa/003/I0183289.dcm\", \"advantech_aifs_aiaa/003/I0183290.dcm\", \"advantech_aifs_aiaa/003/I0183291.dcm\", \"advantech_aifs_aiaa/003/I0183292.dcm\", \"advantech_aifs_aiaa/003/I0183293.dcm\", \"advantech_aifs_aiaa/003/I0183294.dcm\", \"advantech_aifs_aiaa/003/I0183295.dcm\", \"advantech_aifs_aiaa/003/I0183296.dcm\", \"advantech_aifs_aiaa/003/I0183297.dcm\", \"advantech_aifs_aiaa/003/I0183298.dcm\"]}]}"


# curl -X POST $api_url/import --header "Content-Type: application/json" -k \
# --data "{\"endpoint\" : \"$endpoint\", \"access_key\" : \"$access_key\", \"secret_key\" : \"$secret_key\", \"bucket\" : \"$bucket\", 
# \"folder\": [\"advantech_aifs_aiaa/001\", \"advantech_aifs_aiaa/002\"]}"


curl -X POST $api_url/export --header "Content-Type: application/json" -k \
--data "{\"endpoint\" : \"$endpoint\", \"access_key\" : \"$access_key\", \"secret_key\" : \"$secret_key\", \"bucket\" : \"$bucket\", 
\"asset_group\": [{\"category_name\": \"001\", \"files\": [
\"advantech_aifs_aiaa/001/I0182471.dcm\", \"advantech_aifs_aiaa/001/I0182471.json\", \"advantech_aifs_aiaa/001/I0182472.dcm\", \"advantech_aifs_aiaa/001/I0182472.json\", 
\"advantech_aifs_aiaa/001/I0182473.dcm\", \"advantech_aifs_aiaa/001/I0182473.json\", \"advantech_aifs_aiaa/001/I0182474.dcm\", \"advantech_aifs_aiaa/001/I0182474.json\",
\"advantech_aifs_aiaa/001/I0182475.dcm\", \"advantech_aifs_aiaa/001/I0182475.json\", \"advantech_aifs_aiaa/001/I0182476.dcm\", \"advantech_aifs_aiaa/001/I0182476.json\", 
\"advantech_aifs_aiaa/001/I0182477.dcm\", \"advantech_aifs_aiaa/001/I0182477.json\", \"advantech_aifs_aiaa/001/I0182478.dcm\", \"advantech_aifs_aiaa/001/I0182478.json\", 
\"advantech_aifs_aiaa/001/I0182479.dcm\", \"advantech_aifs_aiaa/001/I0182479.json\", \"advantech_aifs_aiaa/001/I0182480.dcm\", \"advantech_aifs_aiaa/001/I0182480.json\", 
\"advantech_aifs_aiaa/001/I0182481.dcm\", \"advantech_aifs_aiaa/001/I0182481.json\", \"advantech_aifs_aiaa/001/I0182482.dcm\", \"advantech_aifs_aiaa/001/I0182482.json\", 
\"advantech_aifs_aiaa/001/I0182483.dcm\", \"advantech_aifs_aiaa/001/I0182483.json\", \"advantech_aifs_aiaa/001/I0182484.dcm\", \"advantech_aifs_aiaa/001/I0182484.json\", 
\"advantech_aifs_aiaa/001/I0182485.dcm\", \"advantech_aifs_aiaa/001/I0182485.json\", \"advantech_aifs_aiaa/001/I0182486.dcm\", \"advantech_aifs_aiaa/001/I0182486.json\", 
\"advantech_aifs_aiaa/001/I0182487.dcm\", \"advantech_aifs_aiaa/001/I0182487.json\", \"advantech_aifs_aiaa/001/I0182488.dcm\", \"advantech_aifs_aiaa/001/I0182488.json\", 
\"advantech_aifs_aiaa/001/I0182489.dcm\", \"advantech_aifs_aiaa/001/I0182489.json\", \"advantech_aifs_aiaa/001/I0182490.dcm\", \"advantech_aifs_aiaa/001/I0182490.json\", 
\"advantech_aifs_aiaa/001/I0182491.dcm\", \"advantech_aifs_aiaa/001/I0182491.json\", \"advantech_aifs_aiaa/001/I0182492.dcm\", \"advantech_aifs_aiaa/001/I0182492.json\", 
\"advantech_aifs_aiaa/001/I0182493.dcm\", \"advantech_aifs_aiaa/001/I0182493.json\", \"advantech_aifs_aiaa/001/I0182494.dcm\", \"advantech_aifs_aiaa/001/I0182494.json\", 
\"advantech_aifs_aiaa/001/I0182495.dcm\", \"advantech_aifs_aiaa/001/I0182495.json\", \"advantech_aifs_aiaa/001/I0182496.dcm\", \"advantech_aifs_aiaa/001/I0182496.json\", 
\"advantech_aifs_aiaa/001/I0182497.dcm\", \"advantech_aifs_aiaa/001/I0182497.json\", \"advantech_aifs_aiaa/001/I0182498.dcm\", \"advantech_aifs_aiaa/001/I0182498.json\", 
\"advantech_aifs_aiaa/001/I0182499.dcm\", \"advantech_aifs_aiaa/001/I0182499.json\", \"advantech_aifs_aiaa/001/I0182500.dcm\", \"advantech_aifs_aiaa/001/I0182500.json\", 
\"advantech_aifs_aiaa/001/I0182501.dcm\", \"advantech_aifs_aiaa/001/I0182501.json\", \"advantech_aifs_aiaa/001/I0182502.dcm\", \"advantech_aifs_aiaa/001/I0182502.json\", 
\"advantech_aifs_aiaa/001/I0182503.dcm\", \"advantech_aifs_aiaa/001/I0182503.json\", \"advantech_aifs_aiaa/001/I0182504.dcm\", \"advantech_aifs_aiaa/001/I0182504.json\", 
\"advantech_aifs_aiaa/001/I0182505.dcm\", \"advantech_aifs_aiaa/001/I0182505.json\", \"advantech_aifs_aiaa/001/I0182506.dcm\", \"advantech_aifs_aiaa/001/I0182506.json\", 
\"advantech_aifs_aiaa/001/I0182507.dcm\", \"advantech_aifs_aiaa/001/I0182507.json\", \"advantech_aifs_aiaa/001/I0182508.dcm\", \"advantech_aifs_aiaa/001/I0182508.json\"
]}, 
{\"category_name\": \"002\", \"files\": [
\"advantech_aifs_aiaa/002/I0182862.dcm\", \"advantech_aifs_aiaa/002/I0182862.json\", \"advantech_aifs_aiaa/002/I0182863.dcm\", \"advantech_aifs_aiaa/002/I0182863.json\", 
\"advantech_aifs_aiaa/002/I0182864.dcm\", \"advantech_aifs_aiaa/002/I0182864.json\", \"advantech_aifs_aiaa/002/I0182865.dcm\", \"advantech_aifs_aiaa/002/I0182865.json\", 
\"advantech_aifs_aiaa/002/I0182866.dcm\", \"advantech_aifs_aiaa/002/I0182866.json\", \"advantech_aifs_aiaa/002/I0182867.dcm\", \"advantech_aifs_aiaa/002/I0182867.json\", 
\"advantech_aifs_aiaa/002/I0182868.dcm\", \"advantech_aifs_aiaa/002/I0182868.json\", \"advantech_aifs_aiaa/002/I0182869.dcm\", \"advantech_aifs_aiaa/002/I0182869.json\", 
\"advantech_aifs_aiaa/002/I0182870.dcm\", \"advantech_aifs_aiaa/002/I0182870.json\", \"advantech_aifs_aiaa/002/I0182871.dcm\", \"advantech_aifs_aiaa/002/I0182871.json\", 
\"advantech_aifs_aiaa/002/I0182872.dcm\", \"advantech_aifs_aiaa/002/I0182872.json\", \"advantech_aifs_aiaa/002/I0182873.dcm\", \"advantech_aifs_aiaa/002/I0182873.json\", 
\"advantech_aifs_aiaa/002/I0182874.dcm\", \"advantech_aifs_aiaa/002/I0182874.json\", \"advantech_aifs_aiaa/002/I0182875.dcm\", \"advantech_aifs_aiaa/002/I0182875.json\", 
\"advantech_aifs_aiaa/002/I0182876.dcm\", \"advantech_aifs_aiaa/002/I0182876.json\", \"advantech_aifs_aiaa/002/I0182877.dcm\", \"advantech_aifs_aiaa/002/I0182877.json\", 
\"advantech_aifs_aiaa/002/I0182878.dcm\", \"advantech_aifs_aiaa/002/I0182878.json\", \"advantech_aifs_aiaa/002/I0182879.dcm\", \"advantech_aifs_aiaa/002/I0182879.json\", 
\"advantech_aifs_aiaa/002/I0182880.dcm\", \"advantech_aifs_aiaa/002/I0182880.json\", \"advantech_aifs_aiaa/002/I0182881.dcm\", \"advantech_aifs_aiaa/002/I0182881.json\", 
\"advantech_aifs_aiaa/002/I0182882.dcm\", \"advantech_aifs_aiaa/002/I0182882.json\", \"advantech_aifs_aiaa/002/I0182883.dcm\", \"advantech_aifs_aiaa/002/I0182883.json\", 
\"advantech_aifs_aiaa/002/I0182884.dcm\", \"advantech_aifs_aiaa/002/I0182884.json\", \"advantech_aifs_aiaa/002/I0182885.dcm\", \"advantech_aifs_aiaa/002/I0182885.json\", 
\"advantech_aifs_aiaa/002/I0182886.dcm\", \"advantech_aifs_aiaa/002/I0182886.json\", \"advantech_aifs_aiaa/002/I0182887.dcm\", \"advantech_aifs_aiaa/002/I0182887.json\", 
\"advantech_aifs_aiaa/002/I0182888.dcm\", \"advantech_aifs_aiaa/002/I0182888.json\", \"advantech_aifs_aiaa/002/I0182889.dcm\", \"advantech_aifs_aiaa/002/I0182889.json\", 
\"advantech_aifs_aiaa/002/I0182890.dcm\", \"advantech_aifs_aiaa/002/I0182890.json\", \"advantech_aifs_aiaa/002/I0182891.dcm\", \"advantech_aifs_aiaa/002/I0182891.json\", 
\"advantech_aifs_aiaa/002/I0182892.dcm\", \"advantech_aifs_aiaa/002/I0182892.json\", \"advantech_aifs_aiaa/002/I0182893.dcm\", \"advantech_aifs_aiaa/002/I0182893.json\", 
\"advantech_aifs_aiaa/002/I0182894.dcm\", \"advantech_aifs_aiaa/002/I0182894.json\", \"advantech_aifs_aiaa/002/I0182895.dcm\", \"advantech_aifs_aiaa/002/I0182895.json\", 
\"advantech_aifs_aiaa/002/I0182896.dcm\", \"advantech_aifs_aiaa/002/I0182896.json\", \"advantech_aifs_aiaa/002/I0182897.dcm\", \"advantech_aifs_aiaa/002/I0182897.json\", 
\"advantech_aifs_aiaa/002/I0182898.dcm\", \"advantech_aifs_aiaa/002/I0182898.json\", \"advantech_aifs_aiaa/002/I0182899.dcm\", \"advantech_aifs_aiaa/002/I0182899.json\", 
\"advantech_aifs_aiaa/002/I0182900.dcm\", \"advantech_aifs_aiaa/002/I0182900.json\", \"advantech_aifs_aiaa/002/I0182901.dcm\", \"advantech_aifs_aiaa/002/I0182901.json\", 
\"advantech_aifs_aiaa/002/I0182902.dcm\", \"advantech_aifs_aiaa/002/I0182902.json\", \"advantech_aifs_aiaa/002/I0182903.dcm\", \"advantech_aifs_aiaa/002/I0182903.json\"
]}
]}"