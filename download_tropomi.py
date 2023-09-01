import math
import os
import sys
import json
import time
import zipfile

from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import oauthlib
from oauthlib.oauth2 import BackendApplicationClient
from requests.packages.urllib3.exceptions import InsecureRequestWarning
from requests_oauthlib import OAuth2Session
from urllib.parse import urlencode, quote

import shelve
import logging
from datetime import timedelta, date, datetime, timezone
from shapely import wkt, geometry
from dateutil.parser import parse
import numpy
import requests
from requests.auth import HTTPBasicAuth

import config
from util import init_logging

if len(sys.argv) < 2 or not sys.argv[1] == "--no-volcview":
    from VolcView import main as sendVolcView

SHOW_PROGRESS = True

auth = HTTPBasicAuth('s5pguest', 's5pguest')


def sentinelhub_compliance_hook(response):
    response.raise_for_status()
    return response


def auth_keycloak():
    token_path = os.path.join(os.path.dirname(__file__), 'ds_auth', 'keycloak_token.jwt')
    token = None
    refresh = False
    if os.path.exists(token_path):
        with open(token_path, 'r') as f:
            token = json.load(f)
        try:
            expires = datetime.utcfromtimestamp(token['expires_at'])
        except KeyError:
            expires = datetime.min

        if expires < datetime.utcnow():
            #token has expired. See if we can refresh
            try:
                refresh_expires = datetime.utcfromtimestamp(token['refresh_expires_at'])
            except KeyError:
                refresh_expires = datetime.min #can't get the refresh expiration date

            if refresh_expires > datetime.utcnow():
                refresh = True
            else:
                token = None

    if token is None or refresh:
        # Get a new token
        data = {
            "client_id": "cdse-public",
        }

        if refresh:
            data['grant_type'] = "refresh_token",
            data['refresh_token'] = token['refresh_token']
        else:
            data['grant_type'] = "password"
            data['username'] = config.SH_EMAIL,
            data['password'] = config.SH_PASSWORD

        try:
            r = requests.post("https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
                              data=data,
                              )
            r.raise_for_status()
        except Exception as e:
            raise Exception(
                f"Keycloak token creation failed. Reponse from the server was: {r.json()}"
            )

        token = r.json()
        # Figure out when it expires
        # Use the response header time as the issued time
        token_time = datetime.utcnow().replace(tzinfo = timezone.utc)
        expires_time: datetime = token_time + timedelta(seconds = int(token['expires_in']))
        refresh_expires = token_time + timedelta(seconds = int(token['refresh_expires_in']))

        token['expires_at'] = expires_time.timestamp()
        token['refresh_expires_at'] = refresh_expires.timestamp()

        with open(token_path, 'w') as f:
            json.dump(token, f)

    session = requests.Session()
    session.headers.update({'Authorization': f'Bearer {token["access_token"]}'})

    return session


def auth_sentinelhub():
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

    session.register_compliance_hook("access_token_response", sentinelhub_compliance_hook)

    return session


def download_part(url, start, end, output):
    downloaded_size = 0
    retry_count = 0
    full_size = end - start
    while True:
        with open(output, 'ab') as part_file:
            range_start = start+part_file.tell()
            range_size = end - range_start
            if range_size <= 0:
                break #Segment complete

            logging.debug(f"Requested range {range_start} - {end}. Size: {end-range_start}")
            resume_header = {'Range': f'bytes={range_start}-{end}', }
            download_request = requests.get(url, stream=True,
                                            headers=resume_header, auth=auth, timeout = 30)
            print(f"Begining download with response code: {download_request.status_code}")
            if download_request.status_code not in (200, 206):
                retry_count += 1
                if retry_count > 10:
                    logging.error("Too many retries. Giving up.")
                    raise FileNotFoundError("Unable to retrieve file part")
                logging.error("Error downloading part %s. Incorrect status code %i. Trying agian in 30 seconds",
                              output, download_request.status_code)
                time.sleep(30)
                continue

            for chunk in download_request.iter_content(chunk_size=4096):  # Try 4K chunk size
                downloaded_size += len(chunk)
                percent_complete = round((downloaded_size / full_size) * 10000, 2)
                if SHOW_PROGRESS and percent_complete % 1 == 0:
                    print(f" {percent_complete / 100}% ", end='\r')
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


def download_sentinelhub(filename, uuid):
    session = auth_keycloak()

    url = f"http://catalogue.dataspace.copernicus.eu/odata/v1/Products({uuid})/$value"
    response = session.get(url, allow_redirects=False)
    while response.status_code in (301, 302, 303, 307):
        url = response.headers['Location']
        response = session.get(url, allow_redirects=False)

    requests.urllib3.disable_warnings(InsecureRequestWarning)
    response = session.get(url, verify = False, allow_redirects = True, stream = True)
    full_size = int(response.headers['Content-Length'])
    downloaded_size = 0
    file = BytesIO()
    last_percent = 0
    for chunk in response.iter_content(chunk_size = 4096):
        downloaded_size += len(chunk)
        percent_complete = int(round((downloaded_size / full_size) * 100))
        if SHOW_PROGRESS and percent_complete != last_percent:
            last_percent = percent_complete
            print(f" {int(round(percent_complete))}% ", end='\r')
        file.write(chunk)

    # Decompress the file
    try:
        with zipfile.ZipFile(file) as f:
            datafile = next((x for x in f.namelist() if x.endswith('.nc')))
            logging.info(f"Extracting {datafile} from archive")
            with f.open(datafile) as df, open(filename + ".download", 'wb') as sf:
                sf.write(df.read())
    except zipfile.BadZipFile as e:
        logging.error(f"Unable to decompress file {filename}. Will try again later. ({e})")


def download_file(file_name, uuid):
    DOWNLOAD_URL = f"https://s5phub.copernicus.eu/dhus/odata/v1/Products('{uuid}')/$value"

    # Request just enough of the file to get the content-length header
    # For some reason this information is not included in a HEAD request.
    #downloaded_size = 0
    download_request = requests.get(DOWNLOAD_URL, stream=True, auth=auth)
    total_size = int(download_request.headers['content-length'])
    # with open(f"{file_name}.download", 'ab') as download_file:

        # for chunk in download_request.iter_content(chunk_size=4096):  # Try 4K chunk size
            # downloaded_size += len(chunk)
            # percent_complete = round((downloaded_size / total_size) * 100, 2)
            # if SHOW_PROGRESS and percent_complete % 1 == 0:
                # print(f" {percent_complete}% ", end='\r')
            # part_file.write(chunk)


    download_request.close()

    workers = 3
    chunk_size = math.ceil(total_size / workers)
    start_byte = 0
    end_byte = 0
    chunk_num = -1
    with ThreadPoolExecutor(max_workers = workers) as executor:
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

    logging.debug("Sending request")
    results = requests.get(URL, auth=auth, headers = headers)
    if results.status_code != 200:
        logging.error(f"An error occured while searching: %i\n%s",
                      results.status_code, results.text)
        return None, results.status_code

    logging.debug("Request received. Parsing JSON")
    results_object = results.json()
    return results_object['products'], 200


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
    features = []
    while not done:
        try:
            results = session.post(SEARCH_URL, json = search_params)
        except oauthlib.oauth2.rfc6749.errors.TokenExpiredError:
            logging.error("Token expired during result fetch. Renewing.")
            session = auth_sentinelhub()
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
            logging.info(f"Search contains more results. Fetching from {next_val}")
            search_params['next'] = next_val

        features += results_object['features']

    names = [{'Name': x['id']} for x in features]
    logging.info("Name list fetched and processed. Getting object ID's")

    # We could query this API directly, but using the catalog API first allows us to use a MultiPolygon
    odata_url = 'https://catalogue.dataspace.copernicus.eu/odata/v1/Products/OData.CSC.FilterList'

    num_files = len(names)
    end_idx = num_files
    
    if num_files > 19:
        num_batches = math.ceil(num_files / 19)
        batch_size = math.ceil(num_files / num_batches)
    else:
        num_batches = 1
        batch_size = num_files
        
    logging.info(f"Using {num_batches} batches of size: {batch_size}")
    features = []
    start_idx = 0
    stop_idx = start_idx + batch_size
    if stop_idx > end_idx:
        stop_idx = end_idx
    idx = 1 # Loop counter

    while start_idx < end_idx:
        batch = names[start_idx:stop_idx]
        logging.info(f"Fetching batch {idx} of {num_batches}")
        odata_request = {
            "FilterProducts": batch,
        }

        results = requests.post(odata_url, json = odata_request)
        results_object = results.json()
        features += results_object['value'] #Object ID's to download
        
        # Increment indexes for next batch
        start_idx = stop_idx
        stop_idx = start_idx + batch_size
        if stop_idx > end_idx:
            stop_idx = end_idx
        idx += 1

    features.sort(key = lambda x: x['OriginDate'], reverse = True)
    return features, 200


def download(use_preop: bool = True):
    init_logging()

    # this produces logging output, so don't import it until *after* we set up logging.
    from h5pyimport import import_product

    logging.info("Begining download process on %s", date.today())

    DEST_DIR = "Offline" if config.OFFLINE else "NRTI"

    cache_location = os.path.join(os.path.dirname(__file__), 'downloadCache')
    with shelve.open(cache_location) as cache:
        from_date = cache.get('lastRun', date.today() - timedelta(days=7))

    to_date = date.today() + timedelta(days=1)
    DATE_TO = to_date.strftime("%Y-%m-%d")

    # Look back 1 week (from the last run) to make sure we have everything
    from_date = from_date - timedelta(days=7)
    DATE_FROM = from_date.strftime("%Y-%m-%d")

    ######DEBUG - REMOVE#######
    #DATE_FROM = "2023-08-30"
#    DATE_TO = "2023-07-20T11:00:00Z"
    ###########################

    logging.info(f"Searching for files from {DATE_FROM} to {DATE_TO}")
    try:        
        if use_preop:
            results_object, code = get_file_list(DATE_FROM, DATE_TO)
            download_func = download_file
        else:
            results_object, code = get_file_list_sentinel_hub(DATE_FROM, DATE_TO)
            download_func = download_sentinelhub
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

    # setup import params
    for idx, product in enumerate(results_object):
        if use_preop:
            uuid = product['uuid']
            footprint = wkt.loads(product['wkt'])
            identifier = product['identifier']
        else:
            uuid = product['Id']
            footprint = geometry.shape(product['GeoFootprint'])
            identifier = product['Name'].replace('.nc', '')

        covered_volcs = [footprint.contains(x) for x in volc_points]
        covered_volcs = [x['name'] for x in volcanos[covered_volcs]]

        id_parts = [x for x in identifier.split('_') if x]
        filetime = parse(id_parts[4] + "z")
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

        download_func(file_name, uuid)

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
    download(use_preop = False)

