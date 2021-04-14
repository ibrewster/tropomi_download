import logging

####################
# Logging config
####################
LOG_FILE = '/var/log/TROPOMI-Download.log'
LOG_FORMAT = '%(asctime)s %(levelname)-8s - %(message)s'

# Log messages at this level or above will output to the screen while running
LOG_CONSOLE_LEVEL = logging.ERROR

# Log messages at this level or above will be saved to the log file
LOG_FILE_LEVEL = logging.INFO

####################
# Type of file to download: NRTI or OFFLINE.
# True = OFFLINE
# False = NRTI
####################
OFFLINE = False

####################
# The directory to which to download files
####################
FILE_BASE = f"/tmp/Transfer/TROPOMIData"

####################
# SECTOR DEFINITIONS
# Download any data files for which at least some portion of the pass
# overlaps one of the below sectors. Any data files for which there
# are only NaN values within the sector(s) covered will be discarded.
####################
# individual sector definitions, defined as
# west,east,south,north or
# longTo,longFrom,latFrom,latTo
alaska_main = [155, 180, 40, 72]
alaska_west = [-180, -140, 40, 72]
marianas = [139.64, 151.36, 11.82, 20.83]
hawaii = [-162, -152, 15, 24]

#  Of the above defined sectors, which ones to actually use
SECTORS = [alaska_main, alaska_west, marianas, hawaii]


####################
# When downloading data files, files are filed by date. An aditional
# hard link to the file can also be created in a seperate directory
# for any volcanos defined below that are covered by the data.
# Ideally, volcano locations should be pulled from a central DB,
# but for now to keep things simple we just use the below data
# structure to define the volcano locations.
####################

VOLCANOS = [
    {'name': 'Kiska',
     'latitude': 52.1031,
     'longitude': 177.6028,
     },
    {'name': 'Little Sitkin',
     'latitude': 51.9531,
     'longitude': 178.5356,
     },
    {'name': 'Semisopochnoi',
     'latitude': 51.9288,
     'longitude': 179.5977,
     },
    {'name': 'Gareloi',
     'latitude': 51.78887,
     'longitude': -178.79368,
     },
    {'name': 'Kanaga',
     'latitude': 51.9242,
     'longitude': -177.1623,
     },
    {'name': 'Great Sitkin',
     'latitude': 52.0765,
     'longitude': -176.1109,
     },
    {'name': 'Korovin',
     'latitude': 52.37934,
     'longitude': -174.1548,
     },
    {'name': 'Cleveland',
     'latitude': 52.8222,
     'longitude': -169.945,
     },
    {'name': 'Okmok',
     'latitude': 53.419,
     'longitude': -168.132,
     },
    {'name': 'Makushin',
     'latitude': 53.88707,
     'longitude': -166.93202,
     },
    {'name': 'Bogoslof',
     'latitude': 53.9272,
     'longitude': -168.0344,
     },
    {'name': 'Akutan',
     'latitude': 54.13308,
     'longitude': -165.98555,
     },
    {'name': 'Westdahl',
     'latitude': 54.5171,
     'longitude': -164.6476,
     },
    {'name': 'Fisher',
     'latitude': 54.6692,
     'longitude': -164.3524,
     },
    {'name': 'Shishaldin',
     'latitude': 54.7554,
     'longitude': -163.9711,
     },
    {'name': 'Pavlof',
     'latitude': 55.4173,
     'longitude': -161.8937,
     },
    {'name': 'Veniaminof',
     'latitude': 56.1979,
     'longitude': -159.3931,
     },
    {'name': 'Aniakchak',
     'latitude': 56.9058,
     'longitude': -158.209,
     },
    {'name': 'Chiginagak',
     'latitude': 57.13348,
     'longitude': -156.99147,
     },
    {'name': 'Ukinrek Maars',
     'latitude': 57.8338,
     'longitude': -156.5139,
     },
    {'name': 'Ugashik-Peulik',
     'latitude': 57.7503,
     'longitude': -156.37,
     },
    {'name': 'Martin',
     'latitude': 58.1692,
     'longitude': -155.3566,
     },
    {'name': 'Mageik',
     'latitude': 58.1946,
     'longitude': -155.2544,
     },
    {'name': 'Trident',
     'latitude': 58.2343,
     'longitude': -155.1026,
     },
    {'name': 'Griggs',
     'latitude': 58.3572,
     'longitude': -155.1037,
     },
    {'name': 'Katmai',
     'latitude': 58.279,
     'longitude': -154.9533,
     },
    {'name': 'Kaguyak',
     'latitude': 58.6113,
     'longitude': -154.0245,
     },
    {'name': 'Fourpeaked',
     'latitude': 58.7703,
     'longitude': -153.6738,
     },
    {'name': 'Douglas',
     'latitude': 58.8596,
     'longitude': -153.5351,
     },
    {'name': 'Augustine',
     'latitude': 59.3626,
     'longitude': -153.435,
     },
    {'name': 'Iliamna',
     'latitude': 60.0319,
     'longitude': -153.0918,
     },
    {'name': 'Redoubt',
     'latitude': 60.4852,
     'longitude': -152.7438,
     },
    {'name': 'Spurr',
     'latitude': 61.2989,
     'longitude': -152.2539,
     },
    {'name': 'Wrangell',
     'latitude': 62.00572,
     'longitude': -144.01935,
     },
    {'name': 'Shiveluch',
     'latitude': 56.6361,
     'longitude': 161.3150,
     },
    {'name': 'Kliuchevskoi',
     'latitude': 56.0556,
     'longitude': 160.6419,
     },
    {'name': 'Bezymianny',
     'latitude': 55.9719,
     'longitude': 160.5953,
     },
    {'name': 'Tolbachik',
     'latitude': 55.8289,
     'longitude': 160.3903,
     },
    {'name': 'Kizimen',
     'latitude': 55.1308,
     'longitude': 160.3200,
     },
    {'name': 'Karymsky',
     'latitude': 54.0486,
     'longitude': 159.1481,
     },
    {'name': 'Zhupanovsky',
     'latitude': 53.3208,
     'longitude': 159.1481,
     },
    {'name': 'Koryaksky',
     'latitude': 53.3208,
     'longitude': 158.7119,
     },
    {'name': 'Avachinsky',
     'latitude': 53.2561,
     'longitude': 158.8361,
     },
    {'name': 'Gorely',
     'latitude': 52.5586,
     'longitude': 158.0303,
     },
    {'name': 'Mutnovsky',
     'latitude': 52.4486,
     'longitude': 158.1964,
     },
    {'name': 'Alaid',
     'latitude': 50.8611,
     'longitude': 155.5650,
     },
    {'name': 'Ebeko',
     'latitude': 50.6856,
     'longitude': 156.0142,
     },
    {'name': 'Chikurachki',
     'latitude': 50.3236,
     'longitude': 155.4611,
     },
    {'name': 'Raikoke',
     'latitude': 48.2911,
     'longitude': 153.2525,
     },
    {'name': 'Sarychev Peak',
     'latitude': 48.0900,
     'longitude': 153.2003,
     },
    {'name': 'Kilauea',
     'latitude': 19.421,
     'longitude': -155.287,
     }
]
