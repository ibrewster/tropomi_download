import os

from download_tropomi import get_file_list_sentinel_hub, download_sentinelhub
from datetime import timedelta, date

import config

from util import init_logging

import numpy

from dateutil.parser import parse
from shapely import wkt, geometry


if __name__ == "__main__":
    init_logging()
    DEST_DIR = "Offline" if config.OFFLINE else "NRTI"

    from_date = date.today() - timedelta(days=1)

    # Go to 00:00 tomorrow to cover through the end of today.
    to_date = date.today() + timedelta(days=1)
    DATE_TO = to_date.strftime("%Y-%m-%d")

    # Look back 1 day to make sure we have everything
    from_date = from_date - timedelta(days=1)
    DATE_FROM = from_date.strftime("%Y-%m-%d")

    results_object, code = get_file_list_sentinel_hub(DATE_FROM, DATE_TO)
    if results_object is None:
        exit(code)

    file_count = len(results_object)
    print("Downloading %d files", file_count)

    volcanos = numpy.asarray(config.VOLCANOS)
    volc_points = [geometry.Point(x['longitude'], x['latitude']) for x in volcanos]

    # setup import params
    for idx, product in enumerate(results_object):
        footprint = geometry.shape(product['geometry'])
        covered_volcs = [footprint.contains(x) for x in volc_points]
        covered_volcs = [x['name'] for x in volcanos[covered_volcs]]

        identifier = product['id']
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
            print("Skipping %s, we already have it.", file_name)
            continue

        download_link = next((x for x in product['links'] if x['rel'] == 'self'))['href']

        print("Downloading %s (%d/%d)", file_name, idx + 1, file_count)

        download_sentinelhub(file_name.replace('.nc', ''), download_link)

        # # Try to import the file to see if we have any valid data
        # print("Checking file for good data")
        # try:
        # # Try importing one of the actual products to make sure it works
        # # properly
        # data = import_product(file_name + ".download",
        # 'valid(SO2_column_number_density);',
        # options = 'so2_column=1km')

        # if data is None or not 'latitude' in data:
        # raise ValueError("No valid so2 data in returned file")

        # # Check to see if there is not NAN data in any of our ranges of interest
        # for west, east, south, north in config.SECTORS:
        # lat_filter = numpy.logical_and(data['latitude'] >= south,
        # data['latitude'] <= north)
        # lon_filter = numpy.logical_and(data['longitude'] >= west,
        # data['longitude'] <= east)
        # good_data = numpy.logical_and(lat_filter, lon_filter).any()
        # if good_data:
        # break
        # else:
        # raise ValueError("No data in file within areas of interest")

        # except ValueError as e:
        # logging.warning("%s in file %s. Perminantly skipping.",
        # str(e), file_name)
        # os.unlink(file_name + ".download")
        # # "touch" a placeholder file so we don't try to download this file again
        # open(file_name + ".skipped", 'a').close()
        # except Exception as e:
        # logging.exception("Failed to load file %s. Will try again later.", file_name)
        # os.unlink(file_name + ".download")
        # # Don't create a skipped file for this, since we don't know what
        # # went wrong. That way we will try again later.
        # else:
        # # Good file, rename it properly
        # os.rename(f"{file_name}.download", f"{file_name}.nc")
        # for volc_name in covered_volcs:
        # dest_dir = os.path.join(volc_dir, volc_name)
        # os.makedirs(dest_dir, exist_ok=True)
        # # Hardlink this file into the proper desitnation folder
        # link_file = os.path.join(dest_dir, f"{product['identifier']}.nc")
        # if os.path.exists(link_file):
        # # There is already a file there, but we want it to be this one, not
        # # whatever exists. So remove the existing file and re-create the hard link.
        # logging.warning("***File already exists in volcano dir. Replacing***")
        # os.unlink(link_file)
        # os.link(f"{file_name}.nc", link_file)
        # # Generate volc view images
        # if len(sys.argv) < 2 or not sys.argv[1] == "--no-volcview":
        # sendVolcView(f"{file_name}.nc")

    with shelve.open(cache_location) as cache:
        cache['lastRun'] = date.today()

    print("Download process complete")
