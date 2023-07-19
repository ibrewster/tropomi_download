import math
import os
import sys
import json

from concurrent.futures import ThreadPoolExecutor
from oauthlib.oauth2 import BackendApplicationClient
from requests_oauthlib import OAuth2Session
from urllib.parse import urlencode, quote

import shelve
import logging
from datetime import timedelta, date, datetime
from shapely import wkt, geometry
from dateutil.parser import parse
import numpy
import requests
from requests.auth import HTTPBasicAuth

import config
from util import init_logging

if len(sys.argv) < 2 or not sys.argv[1] == "--no-volcview":
    from VolcView import main as sendVolcView

SHOW_PROGRESS = False

auth = HTTPBasicAuth('s5pguest', 's5pguest')


def sentinelhub_compliance_hook(response):
    response.raise_for_status()
    return response


def auth_sentinelhub():
    oauth_secret = config.SH_OAUTH_SECRET
    oauth_id = config.SH_OAUTH_ID

    token_path = os.path.join(os.path.dirname(__file__), 'ds_auth', 'token.jwt')
    token = None
    if os.path.exists(token_path):
        with open(token_path, 'r') as f:
            token = json.load(f)

        expires = datetime.utcfromtimestamp(token['expires_at'])
        if expires < datetime.utcnow():
            token = None #token has expired. Get rid of it.

    client = BackendApplicationClient(client_id = oauth_id)
    if token is None:
        session = OAuth2Session(client = client)

        # Get token for the session
        token = session.fetch_token(token_url='https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token',
                                    client_secret=oauth_secret)

        with open(token_path, 'w') as f:
            json.dump(token, f)
    else:
        session = OAuth2Session(client = client, token = token)

    session.register_compliance_hook("access_token_response", sentinelhub_compliance_hook)

    return session


def download_part(url, start, end, output):
    downloaded_size = 0
    retry_count = 0
    full_size = end - start
    while True:
        with open(output, 'ab') as part_file:
            resume_header = {'Range': f'bytes={start+part_file.tell()}-{end}', }
            download_request = requests.get(url, stream=True,
                                            headers=resume_header, auth=auth)

            for chunk in download_request.iter_content(chunk_size=4096):  # Try 4K chunk size
                downloaded_size += len(chunk)
                percent_complete = round((downloaded_size / full_size) * 100, 2)
                if SHOW_PROGRESS and percent_complete % 1 == 0:
                    print(f" {percent_complete}% ", end='\r')
                part_file.write(chunk)

            if downloaded_size >= full_size:
                logging.info("Downloaded file part %s with result %d",
                             output, download_request.status_code)
                download_request.close()
                break
            else:
                retry_count += 1
                if retry_count > 10:
                    logging.error("Too many retries. Giving up.")
                    raise FileNotFoundError("Unable to retrieve file part")

                logging.error("Error downloading part %s. Expected %d bytes, got %d bytes",
                              output, full_size, downloaded_size)
                continue


def download_sentinel_part(url, start, end, output):
    downloaded_size = 0
    retry_count = 0
    full_size = end - start
    while True:
        with open(output, 'ab') as part_file:
            resume_header = {'Range': f'bytes={start+part_file.tell()}-{end}', }
            session = auth_sentinelhub()
            download_request = session.get(url, stream=True,
                                           headers=resume_header)

            for chunk in download_request.iter_content(chunk_size=4096):  # Try 4K chunk size
                downloaded_size += len(chunk)
                percent_complete = round((downloaded_size / full_size) * 100, 2)
                if SHOW_PROGRESS and percent_complete % 1 == 0:
                    print(f" {percent_complete}% ", end='\r')
                part_file.write(chunk)

            if downloaded_size >= full_size:
                logging.info("Downloaded file part %s with result %d",
                             output, download_request.status_code)
                download_request.close()
                break
            else:
                retry_count += 1
                if retry_count > 10:
                    logging.error("Too many retries. Giving up.")
                    raise FileNotFoundError("Unable to retrieve file part")

                logging.error("Error downloading part %s. Expected %d bytes, got %d bytes",
                              output, full_size, downloaded_size)
                continue


def download_sentinelhub(filename, url):
    session = auth_sentinelhub()
    info = session.head(url)
    total_size = int(info.headers['content-length'])

    chunk_size = math.ceil(total_size / 4)
    start_byte = 0
    end_byte = 0
    chunk_num = -1
    with ThreadPoolExecutor(max_workers = 4) as executor:
        while end_byte < total_size:
            chunk_num += 1
            end_byte = start_byte + chunk_size
            if end_byte > total_size:
                end_byte = total_size

            logging.info("Downloading chunk # %i", chunk_num)
            executor.submit(download_sentinel_part, url, start_byte, end_byte,
                            f"{filename}.part{chunk_num}")

            start_byte = end_byte + 1

    # Read in the file parts and write out the full file
    logging.info("Recombining file from parts")
    with open(f"{filename}.download", 'wb') as out_file:
        for i in range(chunk_num + 1):
            chunk_path = f"{filename}.part{i}"
            with open(chunk_path, 'rb') as part_file:
                out_file.write(part_file.read())

            os.remove(chunk_path)


def download_file(file_name, uuid):
    DOWNLOAD_URL = f"https://s5phub.copernicus.eu/dhus/odata/v1/Products('{uuid}')/$value"

    # Request just enough of the file to get the content-length header
    # For some reason this information is not included in a HEAD request.
    download_request = requests.get(DOWNLOAD_URL, stream=True, auth=auth)
    total_size = int(download_request.headers['content-length'])
    download_request.close()

    chunk_size = math.ceil(total_size / 4)
    start_byte = 0
    end_byte = 0
    chunk_num = -1
    with ThreadPoolExecutor(max_workers = 4) as executor:
        while end_byte < total_size:
            chunk_num += 1
            end_byte = start_byte + chunk_size
            if end_byte > total_size:
                end_byte = total_size

            # TODO: Run in parallel
            logging.info("Downloading chunk # %i", chunk_num)
            executor.submit(download_part, DOWNLOAD_URL, start_byte, end_byte,
                            f"{file_name}.part{chunk_num}")

#             download_part(DOWNLOAD_URL, start_byte, end_byte,
#                           f"{file_name}.part{chunk_num}")

            start_byte = end_byte + 1

    # Read in the file parts and write out the full file
    logging.info("Recombining file from parts")
    with open(f"{file_name}.download", 'wb') as out_file:
        for i in range(chunk_num + 1):
            chunk_path = f"{file_name}.part{i}"
            with open(chunk_path, 'rb') as part_file:
                out_file.write(part_file.read())

            os.remove(chunk_path)


def get_file_list(DATE_FROM, DATE_TO):
    SEARCH_URL = "https://s5phub.copernicus.eu/dhus/api/stub/products"

    PROCESS_FILTER = "Offline" if config.OFFLINE else "Near Real Time"
    logging.info("Downloading %s files from %s", PROCESS_FILTER, DATE_FROM)

    footprints = []
    for west, east, south, north in config.SECTORS:
        footprints.append(f'footprint:"Intersects(POLYGON(({west} {south},\
        {east} {south},{east} {north},{west} {north},{west} {south})))"')

    fp = " OR ".join(footprints)

    SEARCH_PARAMS = {'filter': f'( {fp} ) AND \
( beginPosition:[{DATE_FROM}T00:00:00.000Z TO {DATE_TO}T23:59:59.999Z] AND \
endPosition:[{DATE_FROM}T00:00:00.000Z TO {DATE_TO}T23:59:59.999Z] ) AND \
(  (platformname:Sentinel-5 AND producttype:L2__SO2___ AND \
processinglevel:L2 AND processingmode:{PROCESS_FILTER}))',
                     'offset': '0',
                     'limit': '1000',
                     'sortedby': 'beginposition',
                     'order': 'desc'}

    PARAM_STRING = urlencode(SEARCH_PARAMS, safe="/():,[]", quote_via=quote)

    URL = f"{SEARCH_URL}?{PARAM_STRING}"
    headers = {'Accept': 'application/json, text/plain, */*'}

    results = requests.get(URL, auth=auth, headers = headers)
    if results.status_code != 200:
        logging.error(f"An error occured while searching: %i\n%s",
                      results.status_code, results.text)
        return None, results.status_code

    results_object = results.json()
    return results_object, 200


def get_file_list_sentinel_hub(DATE_FROM, DATE_TO):
    if not DATE_TO.endswith('Z'):
        DATE_TO += 'T00:00:00Z'

    if not DATE_FROM.endswith('Z'):
        DATE_FROM += 'T00:00:00Z'

    SEARCH_URL = "https://sh.dataspace.copernicus.eu/api/v1/catalog/1.0.0/search"
    SEARCH_URL = "https://"
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
                'assets',
                'properties.datetime'
            ]
        },
        "filter": f"s5p:timeliness='{PROCESS_FILTER}' and s5p:type='SO2'",
        'intersects': {
            "type": "MultiPolygon",
            "coordinates": footprints,
        },
    }

    session = auth_sentinelhub()
    done = False
    offset = 0
    features = []
    while not done:
        results = session.post(SEARCH_URL, json = search_params)

        if results.status_code != 200:
            logging.error(f"An error occured while searching: %i\n%s",
                          results.status_code, results.text)
            return None, results.status_code

        results_object = results.json()

        next_val = results_object['context'].get('next', -1)
        if next_val < 0:
            done = True
        else:
            search_params['next'] = next_val

        features += results_object['features']

    return features, 200


if __name__ == "__main__":
    init_logging()

    # this produces logging output, so don't import it until *after* we set up logging.
    from h5pyimport import import_product

    logging.info("Begining download on %s", date.today())

    DEST_DIR = "Offline" if config.OFFLINE else "NRTI"

    cache_location = os.path.join(os.path.dirname(__file__), 'downloadCache')
    with shelve.open(cache_location) as cache:
        from_date = cache.get('lastRun', date.today() - timedelta(days=1))

    # DEBUG
    to_date = date.today() + timedelta(days=1)
    DATE_TO = to_date.strftime("%Y-%m-%d")

    # Look back 1 day to make sure we have everything
    from_date = from_date - timedelta(days=1)
    DATE_FROM = from_date.strftime("%Y-%m-%d")

    results_object, code = get_file_list(DATE_FROM, DATE_TO)
    if results_object is None:
        exit(code)

    file_count = len(results_object['products'])
    logging.info("Downloading %d files", file_count)

    volcanos = numpy.asarray(config.VOLCANOS)
    volc_points = [geometry.Point(x['longitude'], x['latitude']) for x in volcanos]

    # setup import params
    for idx, product in enumerate(results_object['products']):
        uuid = product['uuid']
        footprint = wkt.loads(product['wkt'])
        covered_volcs = [footprint.contains(x) for x in volc_points]
        covered_volcs = [x['name'] for x in volcanos[covered_volcs]]

        identifier = product['identifier']
        id_parts = [x for x in identifier.split('_') if x]
        filetime = parse(id_parts[4] + "z")
        year = filetime.strftime("%Y")
        month = filetime.strftime("%m")
        day = filetime.strftime("%d")
        filedate = filetime.strftime('%Y-%m-%d')

        file_dir = os.path.join(config.FILE_BASE, DEST_DIR, year, month, day)
        volc_dir = os.path.join(config.FILE_BASE, DEST_DIR)

        os.makedirs(file_dir, exist_ok=True)
        file_name = os.path.join(file_dir, identifier)
        if os.path.exists(file_name + ".nc") or os.path.exists(file_name + ".skipped"):
            logging.info("Skipping %s, we already have it.", file_name)
            continue

        logging.info("Downloading %s (%d/%d)", file_name, idx + 1, file_count)

        download_file(file_name, uuid)

        # Try to import the file to see if we have any valid data
        logging.info("Checking file for good data")
        try:
            # Try importing one of the actual products to make sure it works
            # properly
            data = import_product(file_name + ".download",
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
            logging.warning("%s in file %s. Perminantly skipping.",
                            str(e), file_name)
            os.unlink(file_name + ".download")
            # "touch" a placeholder file so we don't try to download this file again
            open(file_name + ".skipped", 'a').close()
        except Exception as e:
            logging.exception("Failed to load file %s. Will try again later.", file_name)
            os.unlink(file_name + ".download")
            # Don't create a skipped file for this, since we don't know what
            # went wrong. That way we will try again later.
        else:
            # Good file, rename it properly
            os.rename(f"{file_name}.download", f"{file_name}.nc")
            for volc_name in covered_volcs:
                dest_dir = os.path.join(volc_dir, volc_name)
                os.makedirs(dest_dir, exist_ok=True)
                # Hardlink this file into the proper desitnation folder
                link_file = os.path.join(dest_dir, f"{product['identifier']}.nc")
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
