#!/tropomi_download/bin/python
import logging
import sys
import argparse


sys.path.append('/tropomi_download/')
from VolcView import main as genVolcView

LOG_FILE = '/var/log/omps.log'
logging.basicConfig(filename=LOG_FILE, level=logging.DEBUG)


def str2bool(v):
    if isinstance(v, bool):
        return v
    # else
    isinstance(v, str)
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(f'Boolean value expected. (got: {v})')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a volcview image and upload')
    parser.add_argument('file', type=str)
    parser.add_argument('spawn', type=str2bool, default=True, nargs='?')

    args = parser.parse_args()
    file_name = args.file
    use_spawn = args.spawn
    logging.info("Running with args %s, %s", file_name, str(use_spawn))
    genVolcView(file_name, use_spawn)
    logging.info("Volc view generation completed")
    sys.exit(0)
    print("Called sys.exit")
