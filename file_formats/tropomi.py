from datetime import datetime, timezone
import numpy


__TYPE__ = 'S5P_'


def pointtime_offset(x):
    """
    The pointtime field in TROPOMI data is milliseconds
    since file reference time, convert to seconds.
    """
    return x / 1000


def _file_time(x):
    """
    TROPOMI filetime is seconds since 2010-01-01 00:00:00
    Convert to a real timestamp for use
    """
    return x + datetime(2010, 1, 1, tzinfo=timezone.utc).timestamp()


DEF = {
    'INFO': {
        'ident_attr': {'NAME': 'sensor',
                       'VALUE': b'TROPOMI'},
        'nDims': 3,
        'binRadius': 1.5e4,
        'grid_x_resolution': 6660,  # In meters
        'grid_y_resolution': 6660,
        'file_time': {
            'GROUP': '/PRODUCT',
            'NAME': "time",
            'operation': _file_time,
        },
        'point_time': {'GROUP': "/PRODUCT",
                       'NAME': 'delta_time',
                       'operation': pointtime_offset,
                       },
        'so2_template': {
            'GROUP': '/PRODUCT/SUPPORT_DATA/DETAILED_RESULTS',
            'NAME_PREFIX': 'sulfurdioxide_total_vertical_column_',
            'DEFAULT_GROUP': '/PRODUCT',
            'DEFAULT_NAME': 'sulfurdioxide_total_vertical_column',
            'bin': True,
        },
    },
    'GROUPS': [
        {
            'GROUP': '/PRODUCT',
            'FIELDS': [
                {
                    'NAME': 'latitude',
                    'bin': False,
                },
                {
                    'NAME': 'longitude',
                    'bin': False,
                },
                {
                    'NAME': 'qa_value',
                    'DEST': 'SO2_column_number_density_validity',
                    'bin': False,
                }
            ],
        },
        {
            'GROUP': '/PRODUCT/SUPPORT_DATA/GEOLOCATIONS',
            'FIELDS': [
                {
                    'NAME': 'latitude_bounds',
                    'bin': False,
                },
                {
                    'NAME': 'longitude_bounds',
                    'bin': False,
                },
                {
                    'NAME': 'solar_zenith_angle',
                },
                {
                    'NAME': 'viewing_zenith_angle',
                    'DEST': 'sensor_zenith_angle',
                },
            ],
        },
        {
            'GROUP': '/PRODUCT/SUPPORT_DATA/INPUT_DATA',
            'FIELDS': [
                {
                    'NAME': "cloud_fraction_crb",
                    'DEST': "cloud_fraction",

                }
            ],
        },
        {
            'GROUP': '/PRODUCT/SUPPORT_DATA/DETAILED_RESULTS',
            'FIELDS': [
                {
                    'NAME': 'sulfurdioxide_total_vertical_column_1km',
                    'DEST': 'SO2_number_density_1km',
                },
                {
                    'NAME': 'sulfurdioxide_total_vertical_column_7km',
                    'DEST': 'SO2_number_density_7km',
                },
                {
                    'NAME': 'sulfurdioxide_total_vertical_column_15km',
                    'DEST': 'SO2_number_density_15km',
                }
            ],
        }
    ]
}
