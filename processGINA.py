#!/shared/apps/so2_processing/env/bin/python
from functools import partial
from concurrent.futures import ThreadPoolExecutor

import ginaConfig

from VolcView import main as gen_volc_view

import logging
import os
import shutil

from datetime import datetime, timezone

import paho.mqtt.client as mqtt

logging.basicConfig(filename=ginaConfig.LOG_FILE,
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S',
                    format='%(asctime)s GINA-%(levelname)s: %(message)s'
                    )

SRC_PATH = '/gina_root/upload'
FAILED_DIR = '/gina_root/_failed'

executor = ThreadPoolExecutor(max_workers=3)


def future_complete(filename, dest_file, future):
    try:
        result = future.result()
        logging.info(f"Completed processing of {filename} with return value {result}")
    except Exception:
        logging.exception(f"An exception occurred while processing file {filename}")
        failed_dest=f"{FAILED_DIR}/{filename}"
        os.makedirs(FAILED_DIR, exist_ok=True)
        file_file(filename, failed_dest)
        return

    file_file(filename, dest_file)


def file_file(filename: str, dest: str) -> bool:
    logging.info(
        "Filing %s in %s",
        f"{SRC_PATH}/{filename}",
        dest
    )
    try:
        shutil.move(f"{SRC_PATH}/{filename}", dest)
    except Exception:
        logging.exception(f"Unable to file {filename}")
        return False

    logging.info("Filed: %s", filename)
    return True


def on_message(client, userdata, message):
    """
    Process an incoming MQTT message.

    "message" is an instance of MQTTMessage, a class with members: topic, payload, qos, and retain
    the payload should be the filename to be processed
    """
    logging.info("!!! MESSAGE RECEIVED - HANDLER CALLED !!!")
    try:
        file = message.payload.decode()
        logging.info("Received message to process %s", file)

        file_name = os.path.basename(file)

        if not file_name.endswith('.h5') or not os.path.isfile(file):
            logging.info("Skipping file due to not supported issue")
            return

        if file_name.startswith('V'):
            # VIIRS file
            logging.debug("Detected VIIRS file")
            date_part = file_name[1:14]
            date_format = '%Y%j%H%M%S'
            dest_path = ginaConfig.VIIRS_DEST_DIR
        else:
            # OMPS
            logging.debug("Detected OMPS file")
            file_parts = file_name.split('_')
            date_part = file_parts[3]
            date_format = '%Ym%m%dt%H%M%S'
            dest_path = ginaConfig.OMPS_DEST_DIR

        file_time = datetime.strptime(date_part, date_format)
        file_time = file_time.replace(tzinfo=timezone.utc)
        formatted_date = file_time.strftime('%Y-%m-%d')

        os.makedirs(f"{dest_path}/{formatted_date}", exist_ok=True)
        dest_file = f"{dest_path}/{formatted_date}/{file_name}"

        logging.info("Generating volc view images")
        try:
            future = executor.submit(gen_volc_view, file)
            complete_callback = partial(future_complete, file_name, dest_file)
            future.add_done_callback(complete_callback)
        except Exception:
            logging.exception(f"An exception occurred while processing {file_name}")
    except Exception:
        logging.exception("Unable to process message")


def on_connect(client: mqtt.Client, userdata, flags, rc):
    logging.info(f"Connected to MQTT broker with result code {rc}")
    client.subscribe('GINA', qos=2)
    logging.info("Subscribed to GINA topic")


def on_disconnect(client, userdata, rc):
    if rc != 0:
        logging.warning(f"Unexpected MQTT disconnect. Code: {rc}. Client will auto-reconnect.")
    else:
        logging.info("MQTT disconnected cleanly")


if __name__ == "__main__":
    mqtt_client = mqtt.Client(client_id="gina_processing", clean_session=False)
    mqtt_client.on_message = on_message
    mqtt_client.on_connect = on_connect
    mqtt_client.on_disconnect = on_disconnect
    mqtt_client.connect(ginaConfig.MQTT_SERVER)

    try:
        mqtt_client.loop_forever(retry_first_connection=True)
    except KeyboardInterrupt:
        logging.info("Shutting down...")
    finally:
        mqtt_client.disconnect()
        logging.info("Shutdown complete")
