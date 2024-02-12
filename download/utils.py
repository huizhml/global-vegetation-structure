import os
import json
import ee

def authenticate():
    key_file = os.environ.get('KEY_FILE')
    key_file = key_file or 'keys/private-key.json'
    print('Authenticating from', key_file)
    key = json.load(open(key_file))
    credentials = ee.ServiceAccountCredentials(key['client_email'], key_file)
    ee.Initialize(credentials, url='https://earthengine-highvolume.googleapis.com')
    print('Authenticated Earth Engine successfully') 