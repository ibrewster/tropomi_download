#!/tropomi_download/bin/python
import json
import logging
import os
import sys
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor

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
    args = sys.argv
    print(args)

    logging.info("Got file_info of: %s", str(file_infos))
    gen_procs = []
    with ProcessPoolExecutor() as executor:
        for file in file_infos:
            file_name = file['name']

            if not file_name.endswith('.h5') or not file['exists']:
                continue

            if file_name.startswith('V'):
                # VIIRS file
                date_part = file_name[1:14]
                date_format = '%Y%j%H%M%S'
                DEST_PATH = '/data/viirs'
            else:
                # OMPS
                file_parts = file_name.split('_')
                date_part = file_parts[3]
                date_format = '%Ym%m%dt%H%M%S'
                DEST_PATH = '/omps_data'

            file_time = datetime.strptime(date_part, date_format)
            file_time = file_time.replace(tzinfo=timezone.utc)
            formatted_date = file_time.strftime('%Y-%m-%d')

            os.makedirs(f"{DEST_PATH}/{formatted_date}", exist_ok=True)
            dest_file = f"{DEST_PATH}/{formatted_date}/{file_name}"
            os.rename(f"{SRC_PATH}/{file_name}", dest_file)
            logging.info("Filed: %s %s", file_name, formatted_date)

            logging.info("Generating volc view images")
            future = executor.submit(genVolcView, dest_file)
            gen_procs.append(future)

    logging.info("Waiting for process complete")
    for idx, future in enumerate(gen_procs):
        logging.info("Process %i of %i complete with result: %s", idx + 1,
                     len(gen_procs), str(future.result()))

    logging.info("Complete")
    with open('/tmp/watchmanPython.log', 'a') as logfile:
        logfile.write("Complete\n")
