"""
omps.py - File import definition file for OMPS data files
Copyright © 2020 Alaska Volcano Observatory
Distributed under MIT license. See license.txt for more information
"""

import numpy
from dateutil.parser import parse


__TYPE__ = 'OMPS'


def _omps_pointtime(x):
    return numpy.asarray([parse(date).timestamp()
                          if date.strip()
                          else None
                          for date in x])


def _omps_validity(x):
    return (1 - x) * 100


def _omps_density(x):
    return x / 2241.15


def _get_new_row(row, altrow, idx):
    new_row = row[idx]
    nan_mask = numpy.isnan(new_row)
    if nan_mask.any():
        new_row[nan_mask] = altrow[idx][nan_mask]
    return new_row


def _make_bounds(points):
    if points.min() < -360:
        # We have a fill value. Replace with NaN
        points[points == points.min()] = numpy.NaN

    ll = numpy.roll(points, (-1, 1), axis = (0, 1))
    lr = numpy.roll(points, (-1, - 1), axis = (0, 1))
    ul = numpy.roll(points, (1, 1), axis = (0, 1))
    ur = numpy.roll(points, (1, - 1), axis = (0, 1))

    # Find the diffs
    ur_diff = ur - points
    lr_diff = lr - points
    ul_diff = ul - points
    ll_diff = ll - points

    # Find the first row with some data
    # A row with no data is the same as an edge
    first_row = 0
    for row in range(points.shape[0]):
        if not numpy.isnan(points[row]).all():
            first_row = row
            break

    # fix corner values, since they are weird
    ul_diff[-1, 0] = ul_diff[-1, 1]
    ur_diff[-1, -1] = ur_diff[-1, -2]
    ll_diff[first_row, 0] = ll_diff[first_row, 1]
    lr_diff[first_row, -1] = lr_diff[first_row, -2]

    # Deal with top/bottom edge issues
    ul_diff[first_row] = -1 * _get_new_row(lr_diff, ll_diff, first_row)
    ur_diff[first_row] = -1 * _get_new_row(ll_diff, lr_diff, first_row)

    ll_diff[-1] = -1 * _get_new_row(ur_diff, ul_diff, -1)
    lr_diff[-1] = -1 * _get_new_row(ul_diff, ur_diff, -1)

    # deal with left/right edge issues
    ur_diff[first_row + 1:-1, -1] = -1 * ll_diff[first_row + 1:-1, -1]
    ul_diff[first_row + 1:-1, 0] = -1 * lr_diff[first_row + 1:-1, 0]
    lr_diff[first_row + 1:-1, -1] = -1 * ul_diff[first_row + 1:-1, -1]
    ll_diff[first_row + 1:-1, 0] = -1 * ur_diff[first_row + 1:-1, 0]

    # Deal with dateline crossing diffs
    for diff in (ur_diff, ul_diff, lr_diff, ll_diff):
        diff[diff > 180] -= 360
        diff[diff < -180] += 360

    ur_bounds = points + (ur_diff / 2)
    ul_bounds = points + (ul_diff / 2)
    ll_bounds = points + (ll_diff / 2)
    lr_bounds = points + (lr_diff / 2)

    bounds = numpy.stack([ul_bounds, ur_bounds, lr_bounds, ll_bounds], 2)

    # Fix any out-of-bounds values (crossing dateline)
    bounds[bounds > 180] -= 360
    bounds[bounds < -180] += 360
    return bounds

DEF = {
    'INFO': {
        # The top-level attribute which we can use to identify this file type.
        'ident_attr': {'NAME': 'InstrumentShortName',
                       'VALUE': b'OMPS'},
        'nDims': 2,  # Number of expected dimensions in latitude/longitude products
        # When gridding data, the radius around
        # each grid center point in which to look for data.
        'binRadius': 2e5,
        'grid_x_resolution': 50000,  # In meters
        'grid_y_resolution': 50000,
        'file_time': {},
        'point_time': {
            'GROUP': '/GEOLOCATION_DATA',
            'NAME': 'TimeUTC',
            'operation': _omps_pointtime},
        'so2_template': {
            'GROUP': '/SCIENCE_DATA',
            'NAME_PREFIX': 'ColumnAmountSO2_',
            'DEFAULT_GROUP': '/SCIENCE_DATA',
            'DEFAULT_NAME': 'ColumnAmountSO2_TRM',
            'bin': True,
            'operation': _omps_density
        },
    },
    'GROUPS': [
        {
            'GROUP': '/GEOLOCATION_DATA',
            'FIELDS': [
                {
                    'NAME': 'Latitude',
                    'bin': False,
                    'DEST': 'latitude',
                },
                {
                    'NAME': 'Longitude',
                    'bin': False,
                    'DEST': 'longitude',
                },
                {
                    'NAME': 'Latitude',
                    'bin': False,
                    'operation': _make_bounds,
                    'DEST': 'latitude_bounds',
                },
                {
                    'NAME': 'Longitude',
                    'bin': False,
                    'operation': _make_bounds,
                    'DEST': 'longitude_bounds',
                },
                {
                    'NAME': 'SolarZenithAngle',
                    'DEST': 'solar_zenith_angle',
                },
                {
                    'NAME': 'ViewingZenithAngle',
                    'DEST': 'sensor_zenith_angle',
                }
            ],
        },
        {
            'GROUP': '/SCIENCE_DATA',
            'FIELDS': [
                {
                    'NAME': 'CloudFraction',
                    'DEST': 'cloud_fraction',
                },
                {
                    'NAME': 'Flag_SAA',
                    'DEST': 'SO2_column_number_density_validity',
                    'operation': _omps_validity,
                },
                {
                    'NAME': 'ColumnAmountSO2_PBL',
                    'DEST': 'SO2_number_density_1km',
                    'operation': _omps_density,
                },
                {
                    'NAME': 'ColumnAmountSO2_TRL',
                    'DEST': 'SO2_number_density_3km',
                    'operation': _omps_density,
                },
                {
                    'NAME': 'ColumnAmountSO2_TRM',
                    'DEST': 'SO2_number_density_8km',
                    'operation': _omps_density,
                },
                {
                    'NAME': 'ColumnAmountSO2_TRU',
                    'DEST': 'SO2_number_density_13km',
                    'operation': _omps_density,
                },
                {
                    'NAME': 'ColumnAmountSO2_STL',
                    'DEST': 'SO2_number_density_20km',
                    'operation': _omps_density,
                },
            ],
        }
    ],
}
