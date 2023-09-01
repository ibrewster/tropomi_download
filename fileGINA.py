#!/shared/apps/so2_processing/env/bin/python
import logging
import os
import sys
import time

import ginaConfig

import paho.mqtt.client as mqtt

FILE_PATH = os.path.dirname(__file__)

logging.basicConfig(filename=ginaConfig.LOG_FILE,
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S',
                    format='%(asctime)s GINA-%(levelname)s: %(message)s'
                    )


if __name__ == "__main__":
    files = sys.argv[1:]
    
    client = mqtt.Client()
    client.connect(ginaConfig.MQTT_SERVER)

    logging.info("Got file_info of: %s", str(files))
    for file in files:
        # Make sure file is settled. If not, exit.
        file_size = os.path.getsize(file)
        time.sleep(15)
        new_file_size = os.path.getsize(file)
        if new_file_size != file_size:
            # File is still being transfered. Don't do anything.
            logging.info("File transfer still in process. Not processing")
            exit(1)

        logging.info("Processing file %s", file)
        client.publish('GINA', file)
        continue

    logging.info("Complete")
    with open('/tmp/watchmanPython.log', 'a') as logfile:
        logfile.write("Complete\n")
