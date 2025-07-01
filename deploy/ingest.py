

# ********************** CREATE STAC ITEM **********************
import ipdb
import os
import requests
from azure.identity import AzureCliCredential
from datetime import datetime, timedelta, timezone
import azure.storage.blob
from urllib.parse import urlparse
from rio_stac import create_stac_item
from rasterio.io import MemoryFile
import json
from urllib.parse import urlparse, unquote
from azure.storage.blob import ContainerClient
import pystac
import rasterio
import hydra
from hydra import ConfigStore
from omegaconf import DictConfig
from dataclasses import dataclass


class MPCPro:
    MPCPRO_APP_ID = "https://geocatalog.spatio.azure.com"

    def __init__(self, app_id, api_version, container_name: str = None, storage_account_name: str = None,
                 geocatalog_random_id: str = None, geocatalog_region: str = None, collection_id: str = None,
                 expiration_hours: int = 168):
        self.app_id = app_id
        self.api_version = api_version
        self.collection_id = collection_id
        self.expiration_hours = expiration_hours
        self.container_url = f"https://{storage_account_name}.blob.core.windows.net/{container_name}"
        self.geocatalog_url = f"https://{container_name}.{geocatalog_random_id}.{geocatalog_region}.geocatalog.spatio.azure.com"

    def create_collection(self, collection_title='Global Vegetation Structure Model'):
        collection = {
            "id": self.collection_id,
            "type": "Collection",
            "title": collection_title,
            "description": "Global Vegetation Structure Model",
            "license": "CC-BY-4.0",
            "extent": {
                "spatial": {"bbox": [[-180, -90, 180, 90]]},
                "temporal": {"interval": [[f"{year}-01-01T00:00:00Z", f"{year}-12-31T23:59:59Z"] for year in range(2019, 2023)]},
            },
            "links": [],
            "stac_version": "1.0.0",
            "msft:short_description": "Global Vegetation Structure Model",
        }
        credential = AzureCliCredential()
        access_token = credential.get_token(f'{self.MPCPRO_APP_ID}/.default')
        response = requests.post(
            f"{self.geocatalog_url}/stac/collections",
            json=collection,
            headers={"Authorization": "Bearer " + access_token.token},
            params={"api-version": API_VERSION},
        )
        if response.status_code == 201:
            print(f"Collection {self.collection_id} created successfully")
        else:
            print(f"Failed to create collection {self.collection_id}: {response.status_code} - {response.text}")

    def add_items_to_collection(self, item_list):
        for item in item_list:
            response = requests.post(
                f"{self.geocatalog_url}/stac/collections/{self.collection_id}/items",
                json=item,
                headers={"Authorization": "Bearer " + access_token.token},
                params={"api-version": self.api_version},
            )
            if response.status_code == 201:
                print(f"Item {item['id']} added to collection {self.collection_id} successfully")
            else:
                print(
                    f"Failed to add item {item['id']} to collection {self.collection_id}: {response.status_code} - {response.text}")


    def create_stac_item_from_local_cog(self,path):
        asset_href = ''
        with rasterio.open(path) as dataset:
            stac_item = create_stac_item(
                source=dataset,
                id=path,
                asset_name='data',
                asset_href=path,
                with_proj=True,
                with_raster=True,
                properties={
                    'datetime': None,
                    'raster:bands': [
                        {
                            'nodata': dataset.nodata,
                            'data_type': dataset.dtypes[0],
                            'spatial_resolution': dataset.res[0]
                        }
                    ],
                    'file:size': os.path.getsize(path)
                },
                extensions=[
                    'https://stac-extensions.github.io/file/v2.1.0/schema.json'
                ]
            )
            for band in []:

                stac_item.add_asset(
                    key='data',
                    asset=pystac.Asset(
                        href=path,
                        media_type='image/tiff',
                    )
                )

            return stac_item


def create_stac_item_from_cog(url):
    """
    Create a basic STAC Item for GOES data using rio-stac with proper spatial handling

    Args:
        url (str): URL to the COG file

    Returns:
        pystac.Item: STAC Item with basic metadata and correct spatial information
    """

    # Extract the filename safely from the URL
    parsed_url = urlparse(url)
    path = parsed_url.path  # Gets just the path portion of the URL
    filename = os.path.basename(path)  # Get just the filename part

    # Remove .tif extension if present
    item_id = filename.replace('.tif', '')

    response = requests.get(url, stream=True)
    if response.status_code == 200:
        import ipdb
        ipdb.set_trace()
        with MemoryFile(response.content) as memfile:
            with memfile.open() as dataset:
                # Create base STAC item from rasterio dataset calling create_stac_item from rio_stac
                stac_item = create_stac_item(
                    source=dataset,  # The rasterio dataset object representing the COG file
                    id=item_id,  # Generate a unique ID by extracting the filename without the .tif extension
                    asset_name='data',  # Name of the asset, indicating it contains the primary data
                    asset_href=url,  # URL to the COG file, used as the asset's location
                    with_proj=True,  # Include projection metadata (e.g., CRS, bounding box, etc.)
                    with_raster=True,  # Include raster-specific metadata (e.g., bands, resolution, etc.)
                    properties={
                        'datetime': None,  # Set datetime to None since explicit start/end times may be added later
                        # Add rasterio-specific metadata for the raster bands
                        'raster:bands': [
                            {
                                'nodata': dataset.nodata,  # Value representing no data in the raster
                                'data_type': dataset.dtypes[0],  # Data type of the raster (e.g., uint16)
                                'spatial_resolution': dataset.res[0]  # Spatial resolution of the raster in meters
                            }
                        ],
                        'file:size': len(response.content)  # Size of the file in bytes
                    },
                    extensions=[
                        'https://stac-extensions.github.io/file/v2.1.0/schema.json'  # Add the file extension schema for additional metadata
                    ]
                )

                return stac_item
    else:
        raise Exception(f"Failed to read .tif file: {response.status_code} - {response.text}")


# Function to get the SAS token
def get_sas_token(endpoint):
    response = requests.get(endpoint)
    if response.status_code == 200:
        data = response.json()
        return data.get("token")
    else:
        raise Exception(f"Failed to get SAS token: {response.status_code} - {response.text}")


# ********************** Example usage **********************
# Define Azure Blob Storage parameters
storage_account_name = "vvsm"
container_name = "vsm"
blob_domain = f"https://{storage_account_name}.blob.core.windows.net"

# Set Key Variables Here
# The Planetary Computer Pro App ID. Do not change.
MPCPRO_APP_ID = "https://geocatalog.spatio.azure.com"

# The API version. Do not change.
API_VERSION = "2025-04-30-preview"

# Replace with the URL of the Blob Container
# e.g., "https://youraccount.blob.core.windows.net/yourcontainer"
CONTAINER_URL = f"https://{storage_account_name}.blob.core.windows.net/vsm"

# Replace with the URL of your GeoCatalog Resource
# e.g., "https://yourgeocatalog.randomid.region.geocatalog.spatio.azure.com/"
GEOCATALOG_URL = "https://vsm.hmazaph8h2g5f7dr.westeurope.geocatalog.spatio.azure.com"

# Replace with the desired duration of the token in hours (default: 168 hours = 7 days)
EXPIRATION_HOURS = 168  # e.g., 7 * 24
# Parse the container URL
parsed_url = urlparse(CONTAINER_URL)
account_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
account_name = parsed_url.netloc.split(".")[0]
container_name = parsed_url.path.lstrip("/")

# Login to Azure using the Azure CLI
credential = azure.identity.AzureCliCredential()

# Setup Blob Service Client
with azure.storage.blob.BlobServiceClient(
    account_url=account_url,
    credential=credential,
) as blob_service_client:
    now = datetime.now(timezone.utc).replace(microsecond=0)
    key = blob_service_client.get_user_delegation_key(
        key_start_time=now + timedelta(hours=-1),
        key_expiry_time=now + timedelta(hours=EXPIRATION_HOURS), )

# Generate the SAS token
sas_token = azure.storage.blob.generate_container_sas(
    account_name=account_name,
    container_name=container_name,
    user_delegation_key=key,
    permission=azure.storage.blob.ContainerSasPermissions(
        read=True,
        list=True,
    ),
    start=now + timedelta(hours=-1),
    expiry=now + timedelta(hours=EXPIRATION_HOURS),)


# Create file URL using the first_blob variable that's already defined
container_url = f"{blob_domain}/{container_name}?{sas_token}"
container_client = ContainerClient.from_container_url(container_url)
first_blob = next(container_client.list_blobs())
# Create STAC item for the first blob
file_url = f"{blob_domain}/{container_name}/{first_blob.name}?{sas_token}"
stac_item = create_stac_item_from_cog(file_url)
ipdb.set_trace()

# Print the STAC item as JSON
print(json.dumps(stac_item.to_dict(), indent=2))


@dataclass
class Config:
    MPCPRO_APP_ID = "https://geocatalog.spatio.azure.com"
    API_VERSION = "2025-04-30-preview"
    storage_account_name: str = 'vvsm'
    container_name: str = 'vsm'
    geocatalog_random_id: str = 'hmazaph8h2g5f7dr'
    geocatalog_region: str = 'westeurope'
    expiration_hours: int = 168  # e.g., 7 * 24
    COLLECTION_ID = 'vsm_v1'
    COLLECTION_TITLE = 'Global Vegetation Structure Model'


cs = ConfigStore.instance()
cs.store(name='config', node=Config)


@hydra.main(config_path="config", version_base='1.2')
def main(cfg: DictConfig):
    print(cfg)

    mpcpro = MPCPro(cfg.MPCPRO_APP_ID, cfg.API_VERSION, CONTAINER_URL,
                    GEOCATALOG_URL, cfg.COLLECTION_ID, cfg.expiration_hours)
    mpcpro.create_collection()


if __name__ == "__main__":
    main()
