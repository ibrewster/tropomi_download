#!/tropomi_download/bin/python
import subprocess
import logging
import os
import sys

FILE_PATH = os.path.dirname(__file__)
LOG_FILE = '/var/log/gina.log'
logging.basicConfig(filename=LOG_FILE,
                    level=logging.DEBUG,
                    datefmt='%Y-%m-%d %H:%M:%S',
                    format='%(asctime)s %(levelname)s: %(message)s'
                    )

if __name__ == "__main__":
    file_infos = sys.stdin.read()
    file_exe = os.path.join(FILE_PATH, 'fileOmps2.py')
    proc = subprocess.Popen([file_exe], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    logging.info("Sending data to subprocess %i", proc.pid)
    output = proc.communicate(input=file_infos.encode())
    logging.info("json passed to subprocess")
    exit(1)
