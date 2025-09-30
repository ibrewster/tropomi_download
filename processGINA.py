#!/shared/apps/so2_processing/env/bin/python

import multiprocessing

import ginaConfig

from VolcView import main as genVolcView

import logging
import os
import shutil

from concurrent.futures import ProcessPoolExecutor, process
from datetime import datetime, timezone
from functools import partial

import paho.mqtt.client as mqtt

executor: ProcessPoolExecutor = None

logging.basicConfig(filename=ginaConfig.LOG_FILE,
                    level=logging.INFO,
                    datefmt='%Y-%m-%d %H:%M:%S',
                    format='%(asctime)s GINA-%(levelname)s: %(message)s'
                    )

SRC_PATH = '/gina_root/upload'

def future_complete(filename, future):
    try:
        result = future.result()
    except process.BrokenProcessPool:
        logging.error("!!!!!!!!Process pool died. Re-creating")
        create_process_pool()
        result="Process Pool Restart"
    except Exception:
        logging.exception("An exception occurred while processing file")
        result = future

    logging.info(f"Completed processing of {filename} with return value {result}")


def on_message(client, userdata, message):
    """
    Process an incoming MQTT message.

    message is an instance of MQTTMessage, a class with members topic, payload, qos and retain
    the payload should be the filename to be processed
    """
    global executor

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
            DEST_PATH = ginaConfig.VIIRS_DEST_DIR
        else:
            # OMPS
            logging.debug("Detected OMPS file")
            file_parts = file_name.split('_')
            date_part = file_parts[3]
            date_format = '%Ym%m%dt%H%M%S'
            DEST_PATH = ginaConfig.OMPS_DEST_DIR

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
            shutil.move(f"{SRC_PATH}/{file_name}", dest_file)
        except Exception as e:
            logging.exception(f"Unable to file {file_name}")
            return

        logging.info("Filed: %s %s", file_name, formatted_date)

        logging.info("Generating volc view images")

        # Fire and forget
        complete_callback = partial(future_complete, file_name)
        try:
            future = executor.submit(genVolcView, dest_file, False)
        except process.BrokenProcessPool:
            logging.exception("Process pool broken. Creating a new one and trying again.")
            create_process_pool()
            future = executor.submit(genVolcView, dest_file, False)

        future.add_done_callback(complete_callback)
        # genVolcView(dest_file, False)
    except Exception:
        logging.exception("Unable to process message")
        
def on_connect(client: mqtt.Client, userdata, flags, rc):
    logging.info(f"Connected to MQTT broker with result code {rc}")
    client.subscribe('GINA', qos=2)
    logging.info("Subscribed to GINA topic")

def create_process_pool():
    global executor
    old_executor = executor
    spawn_context = multiprocessing.get_context('spawn')
    executor = ProcessPoolExecutor(max_workers=2, max_tasks_per_child = 1, mp_context = spawn_context)
    
    if old_executor:
        try:
            old_executor.shutdown(wait=False, cancel_futures=False)
        except Exception:
            logging.exception("Error shutting down old executor")
            
def on_disconnect(client, userdata, rc):
    if rc != 0:
        logging.warning(f"Unexpected MQTT disconnect. Code: {rc}. Client will auto-reconnect.")
    else:
        logging.info("MQTT disconnected cleanly")

if __name__ == "__main__":
    create_process_pool()

    client = mqtt.Client(client_id="gina_processing", clean_session=False)
    client.on_message = on_message
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.connect(ginaConfig.MQTT_SERVER)    
    # client.subscribe('GINA')
    
    try:
        client.loop_forever(retry_first_connection = True)
    except KeyboardInterrupt:
        logging.info("Shutting down...")
    finally:
        client.disconnect()
        if executor:
            executor.shutdown(wait=True, cancel_futures=False)
        logging.info("Shutdown complete")
