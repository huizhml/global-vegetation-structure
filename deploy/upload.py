import os
import requests
import os, uuid
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.storage.filedatalake import DataLakeServiceClient

account_url = "https://vvsm.blob.core.windows.net"
account_name = os.getenv('DATALAKE_STORAGE_ACCOUNT_NAME', "")
account_key = os.getenv('DATALAKE_STORAGE_ACCOUNT_KEY', "")

dir_name = '32MQE'

try:
    print("Azure Blob Storage Python quickstart sample")
    default_credential = DefaultAzureCredential()
    # service_client = DataLakeServiceClient(account_url="{}://{}.dfs.core.windows.net".format(
    #     "https",
    #     account_name
    # ), credential=account_key)
    # fs_name = "vvsm"
    # filesystem_client = service_client.create_file_system(file_system=fs_name)
    # directory_client = filesystem_client.create_directory(dir_name)
    container_name = "vsm"
    # Create the BlobServiceClient object
    blob_service_client = BlobServiceClient(account_url, credential=default_credential)
    
    # Create a local directory to hold blob data
    local_path = "/home/ksb781/data/GVS/Deploy/predictions_2017/32MQE/"

    # Create a file in the local data directory to upload and download
    local_file_name = '32MQE_xuyou07n_DEFLATE_7.RH100_Q1_lerc_0_1024.tif'
    upload_file_path = os.path.join(local_path, local_file_name)

    # Create a blob client using the local file name as the name for the blob
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=local_file_name)

    print("\nUploading to Azure Storage as blob:\n\t" + local_file_name)

    # Upload the created file
    with open(file=upload_file_path, mode="rb") as data:
        blob_client.upload_blob(data)

except Exception as ex:
    print('Exception:')
    print(ex)



def upload_to_azure_storage(local_path, remote_path):
    # Create directory if it doesn't exist
    directory = os.path.dirname(remote_path)
    if directory:
        dir_response = requests.put(
            f"https://{os.environ['AZURE_STORAGE_ACCOUNT']}.blob.core.windows.net/{directory}",
            headers={
                "Authorization": f"Bearer {os.environ['AZURE_STORAGE_SAS_TOKEN']}",
                "x-ms-blob-type": "BlockBlob",
                "Content-Length": "0"
            }
        )
        if dir_response.status_code not in [201, 409]:  # 201=Created, 409=Already exists
            raise Exception(f"Failed to create directory: {dir_response.status_code}")

    # Upload the file
    with open(local_path, "rb") as f:
        response = requests.put(
            f"https://{os.environ['AZURE_STORAGE_ACCOUNT']}.blob.core.windows.net/{remote_path}",
            headers={
                "Authorization": f"Bearer {os.environ['AZURE_STORAGE_SAS_TOKEN']}",
                "x-ms-blob-type": "BlockBlob"
            },
            data=f
        )
        
    if response.status_code != 201:
        raise Exception(f"Failed to upload file: {response.status_code}")
    print(f"Successfully uploaded {local_path} to {remote_path}")
    
