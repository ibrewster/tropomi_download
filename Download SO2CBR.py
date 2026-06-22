import json
import logging
import os

from datetime import datetime, timedelta
from urllib.parse import urlparse

import oauthlib
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session


from pystac import Collection
from pystac_client import ItemSearch
import shapely
import boto3

from botocore.config import Config as BotoConfig
from boto3.s3.transfer import TransferConfig

import config

def sentinelhub_compliance_hook(response):
    response.raise_for_status()
    return response

def auth_sentinelhub(token_only=False):
    oauth_secret = config.SH_OAUTH_SECRET
    oauth_id = config.SH_OAUTH_ID

    token_path = os.path.join(os.path.dirname(__file__), 'ds_auth', 'sentinelhub_token.jwt')
    token = None
    if os.path.exists(token_path):
        with open(token_path, 'r') as f:
            token = json.load(f)

        expires = datetime.utcfromtimestamp(token['expires_at'])
        valid_time = round((expires - datetime.utcnow()).total_seconds() / 60, 2)
        logging.info(f"Loaded token will expire in {valid_time} minutes")
        if valid_time < 2:
            token = None #token has expired, or will soon. Get rid of it.

    client = BackendApplicationClient(client_id = oauth_id)
    if token is None:
        logging.info("Fetching new sentinel hub access token")
        session = OAuth2Session(client = client)

        # Get token for the session
        token = session.fetch_token(token_url='https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token',
                                    client_secret=oauth_secret)

        with open(token_path, 'w') as f:
            json.dump(token, f)
    else:
        session = OAuth2Session(client = client, token = token)

    if token_only:
        return token

    session.register_compliance_hook("access_token_response", sentinelhub_compliance_hook)

    return session

def download_s3(url, download_name):
    boto_config = BotoConfig(
        read_timeout=120,
        connect_timeout=60,
        retries={'max_attempts': 10, 'mode': 'standard'}
    )

    # See documentation at https://documentation.dataspace.copernicus.eu/APIs/S3.html
    s3 = boto3.resource(
        's3',
        endpoint_url='https://eodata.dataspace.copernicus.eu',
        aws_access_key_id=config.S3_ACCESS_KEY,
        aws_secret_access_key=config.S3_SECRET_KEY,
        region_name='default',
        config=boto_config
    )

    transfer_config = TransferConfig(
        use_threads=True,
        max_concurrency=5,
        num_download_attempts=10
    )

    s3_bucket = s3.Bucket("eodata")

    files = s3_bucket.objects.filter(Prefix=url)


    print(f"Downloading {url} to {download_name}")
    #session=auth_sentinelhub()
    s3_bucket.download_file(url, download_name,Config=transfer_config)
    print(f"Finished downloading {url} to {download_name}")

def get_alaska_products(date_from, date_to):
    endpoint = 'https://stac.dataspace.copernicus.eu/v1/search'


    footprints = []
    for west, east, south, north in config.SECTORS:
        footprint = [
            [
                [west, south],
                [east, south],
                [east, north],
                [west, north],
                [west, south]
            ]
        ]

        footprints.append(footprint)

    multipoly = shapely.geometry.MultiPolygon(footprints)

    items = ItemSearch(
        endpoint,
        #sortby="-start_datetime",
        collections = ["sentinel-5p-l2-so2-rpro"],
        datetime = [date_from, date_to],
        intersects = multipoly
    ).items()


    items = sorted(items, key = lambda item: item.properties['start_datetime'], reverse = True)
    print("Found",len(items),"to download")

    for item in items:
        # print("Assets:", item.assets)
        filename = f"{item.id}.nc"
        #product_hash = item.properties["hash"]
        filetime = datetime.strptime(item.properties['start_datetime'], '%Y-%m-%dT%H:%M:%S%z')
        year = filetime.strftime("%Y")
        month = filetime.strftime("%m")
        day = filetime.strftime("%d")

        file_dir = os.path.join(config.FILE_BASE, 'COBRA', year, month, day)
        os.makedirs(file_dir, exist_ok = True)
        download_file = os.path.join(file_dir, filename)


        s3_url = item.assets['netcdf'].href
        # print("properties", item.properties)
        #resp = requests.get(item.assets['download'].href, stream = True)
        #full_size = int(resp.headers['Content-Length'])

        if os.path.exists(download_file):
            #file_hash = "md5:" + hashlib.md5(open(download_file, "rb").read(\
            #)).hexdigest()
            #if file_hash == product_hash:
            print("Skipping", download_file, "We already have it!")
            continue

        s3_parsed = urlparse(s3_url, allow_fragments=True)
        path = s3_parsed.path.lstrip('/')
        download_s3(path, download_file)

#        print("Downloading file",download_file, "of size:", full_size)
#        downloaded_size = 0
#        last_percent = 0
#        with open(download_file, 'wb') as file:
#            for chunk in resp.iter_content(chunk_size = 4096):
#                downloaded_size += len(chunk)
#                percent_complete = int(round((downloaded_size / full_size) * 100))
                # if percent_complete != last_percent:
                    # print(f" {int(round(percent_complete))}% ", end='\r')
                    # last_percent = percent_complete
#                file.write(chunk)

        print('-------------')



if __name__ == "__main__":

    dates = [
        "2020-01-07T23:51:08",
        "2020-01-19T23:26:08",
        "2021-03-02T22:33:24",
        "2020-03-16T00:13:50",
        "2021-03-26T23:26:09",
        "2020-04-01T00:13:50",
        "2020-04-19T23:19:48",
        "2020-04-23T23:44:48",
        "2020-04-24T23:24:48",
        "2020-05-05T23:16:39",
        "2018-05-05T22:25:05",
        "2022-05-18T00:46:28",
        "2020-06-04T00:13:18",
        "2020-06-08T00:38:52",
        "2020-06-09T00:18:52",
        "2020-06-17T01:12:32",
        "2020-06-18T00:53:31",
        "2020-06-20T00:13:31",
        "2020-07-06T23:54:08",
        "2020-07-19T23:13:02",
        "2023-07-24T00:49:02",
        "2020-07-26T00:38:16",
        "2023-08-26T00:29:49",
        "2023-09-06T00:22:30",
        "2018-09-25T22:39:11",
        "2023-09-26T22:47:30",
        "2023-10-04T23:37:08",
        "2020-10-26T00:13:39",
    ]

    for date_str in dates:
        date_obj = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S")
        print("Getting data products for", date_obj)
        start_date = date_obj.replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = date_obj.replace(hour=23, minute=59, second=59, microsecond=999999)
        get_alaska_products(start_date, end_date)

    #period = 30
    #end = datetime.now()
    #start = end - timedelta(days = period)
    #while start >= datetime(2022, 7, 19):
    #    print("Getting data products for", start, " - ", end)
    #    get_alaska_products(start, end)
    #    end = start - timedelta(minutes = 1)
    #    start = end -timedelta(days = period)


