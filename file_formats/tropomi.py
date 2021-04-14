import numpy


__TYPE__ = 'S5P_'


def _s5p_validity(x):
    return numpy.round(x * 100)


DEF = {
    'INFO': {
        'nDims': 3,
        'binRadius': 1.5e4,
        'file_time': {
            'GROUP': 'PRODUCT',
            'NAME': "time",
        },
        'point_time': {'GROUP': "PRODUCT",
                       'NAME': 'delta_time'
                       },
        'so2_template': {
            'GROUP': 'PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/',
            'NAME_PREFIX': 'sulfurdioxide_total_vertical_column_',
            'DEFAULT_GROUP': 'PRODUCT',
            'DEFAULT_NAME': 'sulfurdioxide_total_vertical_column',
            'bin': True,
        },
    },
    'GROUPS': [
        {
            'GROUP': 'PRODUCT',
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
                    'operation': _s5p_validity,
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
