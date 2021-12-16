#!/tropomiweb/bin/python
import subprocess
import json
import logging
import os
import sys
from datetime import datetime, timezone
import multiprocessing as mp

#sys.path.append('/tropomiweb/tropomiweb')
#from ImageGen.GenerateVolcView3 import main as genVolcView

SRC_PATH = '/gina_root/upload'
DEST_PATH = '/omps_data'
FILE_PATH = os.path.dirname(__file__)
LOG_FILE = '/var/log/omps.log'
logging.basicConfig(filename=LOG_FILE, level=logging.DEBUG)

if __name__ == "__main__":
    file_infos = json.loads(sys.stdin.read())
    sys.stdin.close()

    for file in file_infos:
        file_name = file['name']
        with open('/tmp/watchmanPython.log', 'a') as logfile:
            logfile.write(f"{file}\n")

        if not file_name.endswith('.h5') or not file['exists']:
            continue

        file_parts = file_name.split('_')
        date_str = file_parts[3][:9]
        formatted_date = datetime.strptime(date_str, '%Ym%m%d').strftime('%Y-%m-%d')
        file_time = datetime.strptime(file_parts[3], '%Ym%m%dt%H%M%S')
        file_time = file_time.replace(tzinfo=timezone.utc)

        os.makedirs(f"{DEST_PATH}/{formatted_date}", exist_ok=True)
        dest_file = f"{DEST_PATH}/{formatted_date}/{file_name}"
        os.rename(f"{SRC_PATH}/{file_name}", dest_file)
        logging.info("Filed: %s %s", file_name, formatted_date)
        print(f"Filed: {file_name} {formatted_date}", file_name, formatted_date)

        logging.info("Generating volc view images")
        gen_exe=os.path.join(FILE_PATH,'genOmps.py')
        logging.info(f"Volcview ImageGen path: {[gen_exe,dest_file,str(file_time),str(False)]}")
        proc=subprocess.Popen([gen_exe,dest_file,str(False)])
        logging.info("Image Gen PID: %s", proc.pid)
        logging.info("Status: %s",proc.poll())
        while True:
            logging.info("Waiting longer...")
            try:
                with open('/tmp/watchmanPython.log', 'a') as logfile:
                    logfile.write("Begining wait loop\n")
                logging.info("Calling wait with timeout 2")
                proc.wait(timeout=2)
                logging.info("Timeout returned!")
                break
            except Exception as e:
                logging.info("Exception caught")
                logging.info(str(e))
                with open('/tmp/watchmanPython.log', 'a') as logfile:
                    logfile.write(str(e)+"\n") 
            except:
                with open('/tmp/watchmanPython.log', 'a') as logfile:
                    logfile.write("Caught Generic Exception\n")
                logging.info("Generic exception caught")
            else:
                break

#        genVolcView(dest_file, file_time, use_spawn=False)
        logging.info("Complete")
        with open('/tmp/watchmanPython.log', 'a') as logfile:
            logfile.write("Complete\n")
