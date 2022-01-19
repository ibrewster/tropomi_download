#!/tropomi_download/bin/python
import logging
import os
import sys
from datetime import datetime, timezone

from VolcView import main as genVolcView

SRC_PATH = '/gina_root/upload'
FILE_PATH = os.path.dirname(__file__)
LOG_FILE = '/var/log/gina.log'

logging.basicConfig(filename=LOG_FILE,
                    level=logging.DEBUG,
                    datefmt='%Y-%m-%d %H:%M:%S',
                    format='%(asctime)s %(levelname)s: %(message)s'
                    )

if __name__ == "__main__":
    files = sys.argv[1:]

    logging.info("Got file_info of: %s", str(files))
    for file in files:
        logging.info("Processing file %s", file)
        file_name = os.path.basename(file)

        if not file_name.endswith('.h5') or not os.path.isfile(file):
            logging.info("Skipping file due to not supported issue")
            continue

        if file_name.startswith('V'):
            # VIIRS file
            logging.debug("Detected VIIRS file")
            date_part = file_name[1:14]
            date_format = '%Y%j%H%M%S'
            DEST_PATH = '/data/viirs_so2'
        else:
            # OMPS
            logging.debug("Detected OMPS file")
            file_parts = file_name.split('_')
            date_part = file_parts[3]
            date_format = '%Ym%m%dt%H%M%S'
            DEST_PATH = '/omps_data'

        file_time = datetime.strptime(date_part, date_format)
        file_time = file_time.replace(tzinfo=timezone.utc)
        formatted_date = file_time.strftime('%Y-%m-%d')

        os.makedirs(f"{DEST_PATH}/{formatted_date}", exist_ok=True)
        dest_file = f"{DEST_PATH}/{formatted_date}/{file_name}"

        logging.info("Filing %s in %s",
                     f"{SRC_PATH}/{file_name}",
                     dest_file
                     )
        try:
            os.rename(f"{SRC_PATH}/{file_name}", dest_file)
        except Exception as e:
            logging.exception(f"Unable to file {file_name}")
            exit(1)

        logging.info("Filed: %s %s", file_name, formatted_date)

        logging.info("Generating volc view images")
        genVolcView(dest_file)

    logging.info("Complete")
    with open('/tmp/watchmanPython.log', 'a') as logfile:
        logfile.write("Complete\n")
