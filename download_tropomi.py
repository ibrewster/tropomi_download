import os
import sys
import json
import zipfile

from io import BytesIO
from urllib.parse import urlparse

import boto3
import oauthlib
from oauthlib.oauth2 import BackendApplicationClient

from requests_oauthlib import OAuth2Session

import shelve
import logging
from datetime import timedelta, date, datetime, timezone
from shapely import geometry
from dateutil.parser import parse
import numpy
import requests

import config
from util import init_logging

if len(sys.argv) < 2 or not sys.argv[1] == "--no-volcview":
    from VolcView import main as sendVolcView

SHOW_PROGRESS = False

def sentinelhub_compliance_hook(response):
    response.raise_for_status()
    return response


# def auth_keycloak():
    # token_path = os.path.join(os.path.dirname(__file__), 'ds_auth', 'keycloak_token.jwt')
    # token = None
    # refresh = False
    # if os.path.exists(token_path):
        # with open(token_path, 'r') as f:
            # token = json.load(f)
        # try:
            # expires = datetime.utcfromtimestamp(token['expires_at'])
        # except KeyError:
            # expires = datetime.min

        # if expires < datetime.utcnow():
            # #token has expired. See if we can refresh
            # try:
                # refresh_expires = datetime.utcfromtimestamp(token['refresh_expires_at'])
            # except KeyError:
                # refresh_expires = datetime.min #can't get the refresh expiration date

            # if refresh_expires > datetime.utcnow():
                # refresh = True
            # else:
                # token = None

    # if token is None or refresh:
        # # Get a new token
        # data = {
            # "client_id": "cdse-public",
        # }

        # if refresh:
            # data['grant_type'] = "refresh_token",
            # data['refresh_token'] = token['refresh_token']
        # else:
            # data['grant_type'] = "password"
            # data['username'] = config.SH_EMAIL,
            # data['password'] = config.SH_PASSWORD

        # try:
            # r = requests.post("https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
                              # data=data,
                              # )
            # r.raise_for_status()
        # except Exception as e:
            # raise Exception(
                # f"Keycloak token creation failed. Reponse from the server was: {r.json()}"
            # )

        # token = r.json()
        # # Figure out when it expires
        # # Use the response header time as the issued time
        # token_time = datetime.utcnow().replace(tzinfo = timezone.utc)
        # expires_time: datetime = token_time + timedelta(seconds = int(token['expires_in']))
        # refresh_expires = token_time + timedelta(seconds = int(token['refresh_expires_in']))

        # token['expires_at'] = expires_time.timestamp()
        # token['refresh_expires_at'] = refresh_expires.timestamp()

        # with open(token_path, 'w') as f:
            # json.dump(token, f)

    # session = requests.Session()
    # session.headers.update({'Authorization': f'Bearer {token["access_token"]}'})

    # return session


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

    
def get_file_list_sentinel_hub(DATE_FROM, DATE_TO):
    if not DATE_TO.endswith('Z'):
        DATE_TO += 'T00:00:00Z'

    if not DATE_FROM.endswith('Z'):
        DATE_FROM += 'T00:00:00Z'

    SEARCH_URL = "https://sh.dataspace.copernicus.eu/api/v1/catalog/1.0.0/search"
    PROCESS_FILTER = "OFFL" if config.OFFLINE else "NRTI"

    logging.info("Downloading %s files from %s", PROCESS_FILTER, DATE_FROM)

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

    search_params = {
        "datetime": f"{DATE_FROM}/{DATE_TO}",
        "collections": ["sentinel-5p-l2"],
        "limit": 100,
        "fields": {
            "exclude": [
                'type',
                'bbox',
            ]
        },
        "filter": f"s5p:timeliness='{PROCESS_FILTER}' and s5p:type='SO2'",
        'intersects': {
            "type": "MultiPolygon",
            "coordinates": footprints,
        },
    }

    session = auth_sentinelhub()
    logging.info("Getting search results")
    done = False
    features = []
    while not done:
        try:
            results = session.post(SEARCH_URL, json = search_params, timeout = 60)
        except oauthlib.oauth2.rfc6749.errors.TokenExpiredError:
            logging.error("Token expired during result fetch. Renewing.")
            session = auth_sentinelhub()
            results = session.post(SEARCH_URL, json = search_params)
        except requests.exceptions.Timeout:
            return None, 408

        if results.status_code != 200:
            logging.error(f"An error occured while searching: %i\n%s",
                          results.status_code, results.text)
            return None, results.status_code

        results_object = results.json()

        next_val = results_object['context'].get('next', -1)
        if next_val < 0:
            done = True
        else:
            logging.info(f"Search contains more results. Fetching from {next_val}")
            search_params['next'] = next_val

        features += results_object['features']

    files = [{
        'Name': x['id'],
        'datetime': x['properties']['datetime'],
        'file_url':x['assets']['data']['href'],
        'GeoFootprint': x['geometry'],
    } for x in features]
    
    files.sort(key=lambda x: x['datetime'], reverse=True)
    return files, 200


def download():
    init_logging()

    # this produces logging output, so don't import it until *after* we set up logging.
    from h5pyimport import import_product

    logging.info("Begining download process on %s", date.today())

    DEST_DIR = "Offline" if config.OFFLINE else "NRTI"

    cache_location = os.path.join(os.path.dirname(__file__), 'downloadCache')
    with shelve.open(cache_location) as cache:
        from_date = cache.get('lastRun', date.today() - timedelta(days=4))

    to_date = date.today() + timedelta(days=2)
    DATE_TO = to_date.strftime("%Y-%m-%d")

    # Look back 1 week (from the last run) to make sure we have everything
    from_date = from_date - timedelta(days=4)
    DATE_FROM = from_date.strftime("%Y-%m-%d")

    ######DEBUG - REMOVE#######
    #DATE_FROM = "2024-03-05"
    #DATE_TO = "2023-07-20T11:00:00Z"
    ###########################

    logging.info(f"Searching for files from {DATE_FROM} to {DATE_TO}")
    try:
        results_object, code = get_file_list_sentinel_hub(DATE_FROM, DATE_TO)
    except Exception as e:
        logging.exception(f"Unable to perform search for files: {e}")
        exit(-5)

    if results_object is None:
        logging.error(f"Search failed with error code: {code}. Giving up for now.")
        exit(code)

    file_count = len(results_object)
    logging.info("Downloading %d files", file_count)

    volcanos = numpy.asarray(config.VOLCANOS)
    volc_points = [geometry.Point(x['longitude'], x['latitude']) for x in volcanos]

    UPDATE_FILE = os.path.join(config.FILE_BASE, DEST_DIR, 'LAST_UPDATE_MARKER.txt')
    
    s3 = boto3.resource(
        's3',
        endpoint_url='https://eodata.dataspace.copernicus.eu',
        aws_access_key_id=config.S3_ACCESS_KEY,
        aws_secret_access_key=config.S3_SECRET_KEY,
        region_name='default'
    )
    s3_bucket = s3.Bucket("eodata")    
    
    for idx, product in enumerate(results_object):
        footprint = geometry.shape(product['GeoFootprint'])
        identifier = product['Name'].replace('.nc', '')

        covered_volcs = [footprint.contains(x) for x in volc_points]
        covered_volcs = [x['name'] for x in volcanos[covered_volcs]]

        #id_parts = [x for x in identifier.split('_') if x]
        filetime = parse(product['datetime'])
        year = filetime.strftime("%Y")
        month = filetime.strftime("%m")
        day = filetime.strftime("%d")

        file_dir = os.path.join(config.FILE_BASE, DEST_DIR, year, month, day)
        volc_dir = os.path.join(config.FILE_BASE, DEST_DIR)

        os.makedirs(file_dir, exist_ok=True)
        file_name = os.path.join(file_dir, identifier)
        if os.path.exists(file_name + ".nc") or os.path.exists(file_name + ".skipped"):
            logging.info("Skipping %s, we already have it.", file_name)
            continue

        logging.info("Downloading %s (%d/%d)", file_name, idx + 1, file_count)

        s3_url = product['file_url']
        s3_parsed = urlparse(s3_url, allow_fragments=True)
        key = s3_parsed.path.lstrip('/')
        s3_file = f"{key}/{product['Name']}"
        download_name = file_name + ".download"
        try:
            s3_bucket.download_file(s3_file, download_name)
        except Exception as e:
            logging.exception(f"Unable to download file: {e}")
            continue

        # Try to import the file to see if we have any valid data
        logging.info("Checking file for good data")
        try:
            # Try importing one of the actual products to make sure it works
            # properly
            data = import_product(download_name,
                                  'valid(SO2_column_number_density);',
                                  options = 'so2_column=1km')

            if data is None or not 'latitude' in data:
                raise ValueError("No valid so2 data in returned file")

            # Check to see if there is not NAN data in any of our ranges of interest
            for west, east, south, north in config.SECTORS:
                lat_filter = numpy.logical_and(data['latitude'] >= south,
                                               data['latitude'] <= north)
                lon_filter = numpy.logical_and(data['longitude'] >= west,
                                               data['longitude'] <= east)
                good_data = numpy.logical_and(lat_filter, lon_filter).any()
                if good_data:
                    break
            else:
                raise ValueError("No data in file within areas of interest")

        except ValueError as e:
            logging.warning("%s in file %s. Permanently skipping.",
                            str(e), file_name)
            os.unlink(file_name + ".download")
            # "touch" a placeholder file so we don't try to download this file again
            open(file_name + ".skipped", 'a').close()
            open(UPDATE_FILE, 'w').close()
        except Exception as e:
            logging.exception("Failed to load file %s. Will try again later.", file_name)
            try:
                os.unlink(file_name + ".download")
            except FileNotFoundError:
                pass # File doesn't exist. Weird, but that *was* the goal here...
            # Don't create a skipped file for this, since we don't know what
            # went wrong. That way we will try again later.
        else:
            # Good file, rename it properly
            os.rename(f"{file_name}.download", f"{file_name}.nc")
            open(UPDATE_FILE, 'w').close()
            for volc_name in covered_volcs:
                dest_dir = os.path.join(volc_dir, volc_name)
                os.makedirs(dest_dir, exist_ok=True)
                # Hardlink this file into the proper desitnation folder
                link_file = os.path.join(dest_dir, f"{identifier}.nc")
                if os.path.exists(link_file):
                    # There is already a file there, but we want it to be this one, not
                    # whatever exists. So remove the existing file and re-create the hard link.
                    logging.warning("***File already exists in volcano dir. Replacing***")
                    os.unlink(link_file)
                os.link(f"{file_name}.nc", link_file)
            # Generate volc view images
            if len(sys.argv) < 2 or not sys.argv[1] == "--no-volcview":
                sendVolcView(f"{file_name}.nc")

    with shelve.open(cache_location) as cache:
        cache['lastRun'] = date.today()

    logging.info("Download process complete")


if __name__ == "__main__":
    download()

