"""
viirs.py - File import definition file for VIIRS data files
Copyright © 2022 Alaska Volcano Observatory
Distributed under MIT license. See license.txt for more information
"""

import numpy
from datetime import datetime


__TYPE__ = 'V'


def _viirs_pointtime(filename):
    time_info = filename[1:14]
    time = datetime.strptime(time_info, '%Y%j%H%M%S')
    return time.timestamp()


def _viirs_validity(x):
    return 1


def _omps_density(x):
    x[x > 100] = 100
    # convert to "DU"-ish type number. NOT quantitative!!!
    x = x * .2

    # Now convert "DU" to "density"
    return x / 2241.15


def _make_nan(points):
    return numpy.full(points.shape, numpy.nan)


def _get_new_row(row, altrow, idx):
    new_row = row[idx]
    nan_mask = numpy.isnan(new_row)
    if nan_mask.any():
        new_row[nan_mask] = altrow[idx][nan_mask]
    return new_row


def _make_bounds(points):
    """Convert a two-dimensional grid of points to a list of pixel corners"""
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
        'ident_attr': {'NAME': 'NoATTR',
                       'VALUE': None},
        'nDims': 2,  # Number of expected dimensions in latitude/longitude products
        # When gridding data, the radius around
        # each grid center point in which to look for data.
        'binRadius': 2e5,
        'grid_x_resolution': 50000,  # In meters
        'grid_y_resolution': 50000,
        'fillvalue': -999.8,
        'file_time': {},
        'point_time': {
            'GROUP': '/All_Data/VIIRS-SO2-ASH-INDICES-L2_All',
            'NAME': 'SO2index',  # doesn't really matter, as long as it has the right shape
            'from_name': _viirs_pointtime,
        },
        'so2_template': {
            'GROUP': '/All_Data/VIIRS-SO2-ASH-INDICES-L2_All',
            'NAME_PREFIX': '',
            'DEFAULT_GROUP': '/All_Data/VIIRS-SO2-ASH-INDICES-L2_All',
            'DEFAULT_NAME': 'SO2index',
            'bin': True,
            'operation': _omps_density
        },

    },
    'GROUPS': [
        {
            'GROUP': '/All_Data/VIIRS-SO2-ASH-INDICES-L2_All',
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
                    'NAME': 'Sensor_Zenith',
                    'DEST': 'sensor_zenith_angle',
                }
            ],
        }
    ],
}
