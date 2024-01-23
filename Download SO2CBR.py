import os
from datetime import datetime, timedelta
from io import BytesIO
from pystac import Collection
from pystac_client import ItemSearch
import requests
import shapely

import config

def get_alaska_products(DATE_FROM, DATE_TO):
    s5pcoll = Collection.from_file(
        "https://data-portal.s5p-pal.com/cat/sentinel-5p/S5P_L2__SO2CBR/catalog.json"
    )
    endpoint = s5pcoll.get_single_link("search").target


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
        datetime = [DATE_FROM, DATE_TO],
        intersects = multipoly
    ).items()

    items = tuple(items)
    print("Found",len(items),"to download")

    items = sorted(items, key = lambda item: item.properties['start_datetime'], reverse = True)
    for item in items:
        print("Assets:", item.assets)
        filename = item.properties['physical_name']
        filetime = datetime.strptime(item.properties['start_datetime'], '%Y-%m-%dT%H:%M:%S%z')
        year = filetime.strftime("%Y")
        month = filetime.strftime("%m")
        day = filetime.strftime("%d")

        file_dir = os.path.join(config.FILE_BASE, 'COBRA', year, month, day)
        os.makedirs(file_dir, exist_ok = True)
        download_file = os.path.join(file_dir, filename+".download")

        print("properties", item.properties)
        resp = requests.get(item.assets['download'].href, stream = True)
        full_size = int(resp.headers['Content-Length'])

        print("Downloading file",download_file, "of size:", full_size)
        downloaded_size = 0
#        last_percent = 0
        with open(download_file, 'wb') as file:
            for chunk in resp.iter_content(chunk_size = 4096):
                downloaded_size += len(chunk)
#                percent_complete = int(round((downloaded_size / full_size) * 100))
                # if percent_complete != last_percent:
                    # print(f" {int(round(percent_complete))}% ", end='\r')
                    # last_percent = percent_complete
                file.write(chunk)

        print('-------------')



if __name__ == "__main__":
    period = 30
    end = datetime.now()
    start = end - timedelta(days = period)
    while start >= datetime(2022, 7, 19):
        print("Getting data products for", start, " - ", end)
        get_alaska_products(start, end)
        end = start - timedelta(minutes = 1)
        start = end -timedelta(days = period)


