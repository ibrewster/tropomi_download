"""
h5pyimport.py - Functions for loading data from NetCDF files using the coda
    library
Copyright © 2021 Alaska Volcano Observatory
Distributed under MIT license. See license.txt for more information
"""
import gc
import importlib
import logging
import multiprocessing
import multiprocessing.dummy as mpd
import operator as op
import os
import pickle
import queue
#import sys
import threading
import time
import warnings

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from glob import glob
from copy import deepcopy

import h5py
import numpy
import xarray

from pyresample import geometry, create_area_def
from pyresample.bilinear import NumpyBilinearResampler

import util as utils

########### GLOBAL CONSTANTS ###############
_REQUIRED_FIELDS = ('latitude', 'longitude', 'sulfurdioxide_total_vertical_column',
                    'latitude_bounds', 'longitude_bounds', "cloud_fraction",
                    'SO2_column_number_density_validity', 'SO2_number_density', 'SO2_column_number_density')

############################################

def flatten_data(data):
    """
        Take a N-dimensional data structure, and "flatten" it to a
        N-1 dimensional structure

        Parameters
        ----------
        data : dictionary
            A data dictonary, containing one or more data structures to be
            flattened

        Returns
        -------
        data : dictionary
            The dictionary of data, with the flattened data structures
    """
    # Flatten the x-y data into a simple time-series type data
    data = data.stack(time = ["x", "y"]).reset_index('time', drop = True)

    # Get the mean of the data across the "file" dimension
    try:
        data = data.mean('file')
    except ValueError:
        pass  # No file dimension - only one file, no need to average anything.

    # make sure the dimensions are in the correct order
    args = ['time', 'corners']
    if 'layer' in data.dims:
        args.append('layer')
    data = data.transpose(*args)

    return data

def _load_formats():
    # Get a list of .py files from the file_formats directory adjacent to this file.
    # These files contain definitions of how to import data from various file types
    # into the format we expect.
    # File definition files MUST contain a __TYPE__ variable, and a DEF
    # dictionary containing the import parameters (fields, info, etc)
    file_defs = {}

    FORMAT_DIR = os.path.join(os.path.dirname(__file__),
                              'file_formats')
    formats = [x[:-3] for x in os.listdir(FORMAT_DIR)
               if x.endswith('.py')]

    # Load all file formats found in the formats directory
    for fmt in formats:
        module = importlib.import_module(f'file_formats.{fmt}')
        file_defs[module.__TYPE__] = module.DEF

    return file_defs


def _get_pointtimes(file_def, file_xa, h5_file, filepath, data_size, data_shape):
    pt_group = file_def['INFO']['point_time']['GROUP']
    pt_name = file_def['INFO']['point_time']['NAME']
    from_name = file_def['INFO']['point_time'].get('from_name')
    point_time_data = numpy.asarray(h5_file[pt_group][pt_name])
    if from_name:
        filename = os.path.basename(filepath)
        point_time = from_name(filename)
        point_time_data = numpy.full(point_time_data.shape, point_time)

    time_op = file_def['INFO']['point_time'].get('operation')
    if time_op:
        point_time_data = time_op(point_time_data)

    file_time_group = file_def['INFO'].get('file_time', {}).get('GROUP')
    if file_time_group:
        file_time_name = file_def['INFO'].get('file_time', {}).get('NAME')
        file_time_path = f"{file_time_group}/{file_time_name}"

        file_time = h5_file[file_time_path][0]
        file_time_op = file_def['INFO'].get('file_time', {}).get('operation')
        file_time = file_time_op(file_time)

        point_time_data = point_time_data + file_time  # Now an actual timestamp value

    if point_time_data.size != data_size:
        # raise ValueError("Data sizes don't match!")
        num_repeat = data_size / point_time_data.size
        point_time_data = numpy.repeat(point_time_data, num_repeat).reshape(data_shape)

    point_time_data = point_time_data.squeeze()
    file_xa.coords['datetime_start'] = (["y", "x"], point_time_data)


def _load_file_data(file_def, filepath, fields: tuple = _REQUIRED_FIELDS):
    # if fields is None when called, use the minimal list of required fields
    # if fields is None when running, then use all fields.
    if fields is None:
        fields = _REQUIRED_FIELDS
    elif fields == 'ALL':
        fields = None
    # If a list of fields is supplied, make sure latitude is one of them
    # as we use that for other purposes.
    elif 'latitude' not in fields:
        fields = tuple(fields) + ('latitude', )

    file_xa = xarray.Dataset()
    h5_file = h5py.File(filepath, 'r')

    # Find the latitude entry to get the data size
    field = {}  # Make the linter happy by making sure these are defined.
    group = {}
    for group in file_def['GROUPS']:
        for field in group['FIELDS']:
            if field.get('DEST', field['NAME']) == 'latitude':
                break
        else:
            # If we didn't find latitude in the previous group,
            # move on to the next group
            continue

        break

    try:
        if field.get('DEST', field['NAME']) == 'latitude':
            group_name = group['GROUP']
            field_name = field['NAME']
            data_size = h5_file[f"/{group_name}/{field_name}"].size
            data_shape = h5_file[f"/{group_name}/{field_name}"].shape
        else:
            raise KeyError("Latitude not found in file def")
    except KeyError as err:
        # Path does not exist in file. Bad file, apparently
        logging.error(str(err))
        return {}

    # Generate the time array
    _get_pointtimes(file_def, file_xa, h5_file, filepath, data_size,
                    data_shape)

    # Fetch desired data from file
    group_spec = (
        (g['GROUP'], f)
        for g in file_def['GROUPS']
        for f in g['FIELDS']
    )

    spec_fill = file_def['INFO'].get('fillvalue')  # May be none

    for group, spec in group_spec:
        field = spec.get('DEST', spec['NAME'])
        if fields is not None and not field.startswith(fields):
            logging.debug(f"Skipping {field} as it is not in the requested list")
            continue

        path = f"{group}/{spec['NAME']}"
        try:
            field_data = numpy.asarray(h5_file[path]).squeeze()
        except (KeyError, OSError) as e:
            logging.error(f"Unable to load field {path} due to error: {e}")
            continue # Key not in file, or unable to read data.

        fill_value = spec_fill or h5_file[path].fillvalue
        try:
            field_data[field_data == fill_value] = numpy.nan
        except ValueError:
            pass

        isinstance(field_data, numpy.ndarray)
        if 'operation' in spec:
            field_data = spec['operation'](field_data)

        # flat_shape = (-1, *field_data.shape[file_def['INFO']['nDims']:])
        # field_data.shape = flat_shape

        dim = ["y", "x"] if len(field_data.shape) > 1 else ['layer']
        if len(field_data.shape) > 2:
            if field_data.shape[2] == 4:
                dim.append('corners')
            else:
                dim.append('layer')

        if field in ['latitude', 'longitude',
                     'latitude_bounds', 'longitude_bounds']:
            file_xa.coords[field] = (dim, field_data)
        else:
            file_xa[field] = (dim, field_data)

    return file_xa


def _parse_filters(filters):
    result = []
    for filter_str in filters:
        if "(" in filter_str:
            result.append(filter_str)
            continue

        field = ""
        operator = ""
        value = ""
        # Split filter into parts
        section = 0
        for char in filter_str.strip():
            if section == 0:
                if char not in ['>', '<', '=']:
                    field += char
                    continue

                section = 1
                # grab this character for the operator
                operator += char
                continue

            elif section == 1:
                # Working on the operator. NON alpha numeric only
                if char in ['>', '<', '=']:
                    operator += char
                    continue

                section = 2
                # We are now on the value
                value += char
            else: # section == 2
                # remainder of the string is the value. Could short-circuit by using enum,
                # and slicing once we got here. Probably not worth the optimization though.
                value += char

        value = float(value)

        result.append((field, operator, value))
    return result


def get_filetype(file: str) -> tuple[str, dict]:
    FILE_DEFS = _load_formats()
    ftype = fdef = None  # Will be over-written by the for loop

    with h5py.File(file, 'r') as f:
        for ftype, fdef in FILE_DEFS.items():
            try:
                field_defs = [
                    fld_lst.get('GROUP') + "/" + fld.get('NAME')
                    for fld_lst in fdef['GROUPS']
                    for fld in fld_lst['FIELDS']
                ]
            except TypeError:
                field_defs = ['/invalid/field']

            ident_attr = fdef.get('INFO', {}).get('ident_attr', {}).get('NAME')

            # See if we can make a positive identification based on file attributes.
            if ident_attr is not None and ident_attr != "NoATTR":
                ident_val = fdef.get('INFO', {}).get('ident_attr', {}).get('VALUE')
                if f.attrs.get(ident_attr, 'NoValue') == ident_val:
                    logging.info(f"Detected file of type {ftype} based on attribute match")
                    break  # Yay! We found the correct file format!
                else:
                    # ident attrs don't match, this is not the file format we are looking for.
                    continue

            # But maybe we aren't expecting *any* attributes in the file.
            if ident_attr == 'NoATTR':
                # If we specify NoATTR, that means that this file format should have no
                # attributes in the file. If the file *does* have attributes, then, we
                # can positively say it is not this format, and move on.
                if len(f.attrs) != 0:
                    continue

            # At this point, we either have no attributes, and are expecting none, or we don't
            # have an identifying attribute listed. Fall back to checking for fields.
            for field in field_defs:
                if not field in f or field is None:
                    break  # breaks the inner loop, so the else clause does not trigger.
            else:
                # All Fields match! We found our file and can break this outter loop!
                logging.warning(f"Guessing file type of {ftype} based on field matching")
                break

        else:
            logging.error("Unable to identify file type")
            raise TypeError("Unable to identify file type")

    # Return a COPY of the filedef to avoid accidentally changing the original at any point
    return (ftype, deepcopy(fdef))


class NetCDFFile:
    _filter_ops = {
        ">": op.gt,
        ">=": op.ge,
        "==": op.eq,
        "<=": op.le,
        "<": op.lt,
    }

    def __init__(self, files = None):
        self._files = []
        self._file_data = None
        self._file_idxs = []
        self._file_type = None
        self._file_def = None
        if files is not None:
            self.configure(files)

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        del self._file_data

    def configure(self, files):
        """
        PARAMETERS
        ----------
        files : str or list
                If str, the file to import. If list, a list of files to import
        """
        if not isinstance(files, (list, tuple)):
            files = [files]

        # Process wildcards in filenames
        self._files = []
        for filespec in files:
            if "*" in filespec or "?" in filespec:
                self._files += glob(filespec)
            else:
                self._files.append(filespec)

        _, fdef = get_filetype(self._files[0])

        self._file_def = fdef

        self._fields = [
            fld.get('DEST', fld['NAME'])
            for fld_lst in fdef['GROUPS']
            for fld in fld_lst['FIELDS']
        ]

        # Get a list of fields that need binned
        self._to_bin = [
            fld.get('DEST', fld['NAME'])
            for fld_lst in self._file_def['GROUPS']
            for fld in fld_lst['FIELDS']
            if fld.get('bin', True)
        ]

    def import_data(self, filters, options, fields = None):
        """
            Main function of this class, given a list of files, filters, and
            options, import the data from the specified file(s) into a
            pre-defined data structure

            Parameters
            ----------
            filters : str
                A semi-colon delimited list of filters to apply to the data on
                import. Can be either in the form field-bool-value or a special
                function call defined here, such as valid(field_name)
            options : str
                A semi-colon list of file-type specific options defined as
                option=value

            Returns
            -------
            data : xarray
                A xarray of data loaded from the specified file(s) and
                filtered with the specified filter(s)
        """
        if self._file_data:
            del self._file_data
            self._file_data = None
            gc.collect()

        # Parse out the options
        if options is None:
            _options = {}
        elif not isinstance(options, str):
            raise TypeError(f"Options must be a string, not {type(options)}")
        else:
            _options = {x.split("=")[0]: x.split("=")[1] for x in options.split(';') if x}

        so2_column = _options.get('so2_column')

        # Set the path to the proper SO2 product depending on filter entered (if any)
        # get the SO2 template
        template = self._file_def['INFO']['so2_template']

        if so2_column is None:
            so2_group = template['DEFAULT_GROUP']
            so2_name = template['DEFAULT_NAME']
        else:
            so2_group = template['GROUP']
            prefix = template['NAME_PREFIX']
            so2_name = f"{prefix}{so2_column}"

        so2_field = {
            'NAME': so2_name,
            'bin': template.get('bin', True),
            'DEST': 'SO2_column_number_density',
        }

        if 'operation' in template:
            so2_field['operation'] = template['operation']

        # find group index
        for idx, group in enumerate([x['GROUP'] for x in self._file_def['GROUPS']]):
            if group == so2_group:
                break
        else:
            idx = None

        self._file_def['GROUPS'][idx]['FIELDS'].append(so2_field)

        # Parse the filter string (if any)
        if filters is None:
            filters = []
        elif not isinstance(filters, str):
            raise TypeError(f"Filters must be a string, not {type(filters)}")
        else:
            filters = filter(None, filters.split(";"))

        filters = _parse_filters(filters)
        # Check for valid filters
        for filter_item in filters:
            if isinstance(filter_item, str):
                continue  # function filter, handle later

            if not filter_item[0] in tuple(self._fields) + ('datetime_start', ):
                raise ValueError(f"Cannot filter on non-existent variable {filter_item[0]}")

        # Concatenate data from all files into a single data structure
        field_count = len(self._fields)
        total_steps = field_count * len(self._files) + len(self._files) + len(filters)

        # From here on, we need to make sure this thread is told to quit at any point we
        # might exit this function, whether due to error or intent.

        # Get a list of fields required for filters. Will add some bogus entries, but it works
        # Because we simply ask if a field in the file is included in this list, so extra
        # entries don't cause problems.
        if fields is None:
            filter_fields = (x[0] for x in filters)
            fields = tuple(filter_fields) + _REQUIRED_FIELDS

        try:
            load_file_data = partial(_load_file_data, self._file_def, fields = fields)
            num_files = len(self._files)

            if num_files == 1:
                # No sense in doing multiprocessing if there is only one file to load.
                all_data = [load_file_data(self._files[0])]
            else:
                process_count = 3
                # Use multiprocessing if we can, otherwise use a thread pool.
                # Thread pool is about 25% slower in testing, but still faster
                # than non-threaded.
                try:
                    pool = multiprocessing.Pool(process_count)
                except AssertionError:
                    # Use threads (multiprocessing.dummy)
                    pool = mpd.Pool(process_count)

                all_data = pool.imap(load_file_data, self._files)
                pool.close()

            self._file_data = list(all_data)

            if not self._file_data or not self._file_data[0]:
                data_size = 0
            else:
                data_size = sum([x.sizes['x'] * x.sizes['y'] for x in self._file_data])

            if data_size == 0:
                return {}  # No data matching filters in these files

            # We have our data loaded. Apply filters in order provided.
            # We need to apply again here so we can grid, if needed, and do
            # anything that comes after the gridding.
            for filter_ in filters:
                # Two types of filters: function filters like bin_spatial()
                # and data filters like latitude<50
                if isinstance(filter_, str):
                    func, args = filter_.split("(")
                    # remove the closing parenthasis. Could probably do a chop
                    # as well.
                    args = args.replace(")", '')
                    args = args.split(",")
                    func = "_" + func

                    # Will throw an error if func is not defined
                    # better be a function that takes the proper arguments and returns the file_data object
                    getattr(self, func)(*args)
                else:
                    self._apply_filter(*filter_)

        except Exception as e:
            logging.exception("Error when loading file:\n")
            # yes, we want to catch anything, because we need to make sure the
            # thread exits cleanly. No exceptions. Even something like a keyboard interupt
            raise

        if isinstance(self._file_data, (list, tuple)):
            if len(self._file_data) == 1:
                self._file_data = self._file_data[0]
            else:
                raise ValueError("Multiple files found, but not stacking")

        return self._file_data

    def _apply_filter(self, field, operator, value, data = None):
        if data is None:
            data = self._file_data

        if not isinstance(data, (list, tuple)):
            data = [data, ]

        for idx, file in enumerate(data):
            _data_filter = self._filter_ops[operator](file[field], value)

            if _data_filter.all():
                continue  # Not removing any data, no need to go further.

            data[idx] = file.where(_data_filter, drop = True)

        return data

    def _valid(self, key, data = None):
        set_self = False
        if data is None:
            data = self._file_data
            set_self = True

        for idx, file in enumerate(data):
            _data_filter = ~numpy.isnan(file[key])

            if _data_filter.all():
                return  # Not removing any data, no need to go further.

            data[idx] = file.where(_data_filter, drop = True)
        if set_self:
            self._file_data = data

        return data

    def _longitude_range(self, lon_from, lon_to, data = None):
        """Always east to west lat from to lat to"""
        set_self = False
        if data is None:
            data = self._file_data
            set_self = True

        lon_from = float(lon_from)
        lon_to = float(lon_to)

        if lon_from < -180:
            lon_from += 360

        if lon_to > 180:
            lon_to -= 360

        if lon_from > lon_to:
            lon_from = [lon_from, -180]
            lon_to = [180, lon_to]
        else:
            lon_from = [lon_from]
            lon_to = [lon_to]

        for idx, file in enumerate(data):
            filters = []
            longitude_data = file['longitude']
            for start, stop in zip(lon_from, lon_to):
                filters.append((longitude_data >= start) & (longitude_data <= stop))

            if len(filters) > 1:  # if not one, then two.
                _data_filter = numpy.logical_or(*filters)
            else:
                _data_filter = filters[0]

            if _data_filter.all():
                continue

            # Have to do this somewhat cludigly, to make everything work.
            file = file.where(_data_filter)
            for item in file.coords:
                file.coords[item] = file[item].where(_data_filter)

            # drop what we can
            slim_file = file.where(_data_filter, drop = True)

            # and replace the coordinates that were inexplicably removed by that operation
            for item in file.coords:
                slim_file.coords[item] = file[item].where(_data_filter, drop = True)

            data[idx] = slim_file

        if set_self:
            self._file_data = data

        return data

    def _bin_spatial(self, num_lat, lat_from, lat_step, num_lon, lon_from, lon_step,
                     proj=None):
        # Make sure everything is correctly typed
        num_lat = int(num_lat) - 1
        lat_from = float(lat_from)
        lat_step = float(lat_step)
        lat_to = lat_from + (num_lat * lat_step)

        num_lon = int(num_lon) - 1
        lon_from = float(lon_from)
        lon_step = float(lon_step)
        lon_to = lon_from + (num_lon * lon_step)

        # try gridding the data
        area_extent = (lon_from, lat_from,
                       lon_to, lat_to)

        if isinstance(proj, str):
            proj = pickle.loads(bytes.fromhex(proj))

        proj_dict = proj or {'proj': 'lonlat', 'over': True, 'preserve_units': False, }
        target_def = create_area_def("Binned", proj_dict, area_extent=area_extent,
                                     shape=(num_lat, num_lon))

        target_lons, target_lats = target_def.get_lonlats()

        # Fix "out-of-bounds" lons
        if proj is None:
            target_lons[target_lons < -180] = 180 - (-1 * (target_lons[target_lons < -180] + 180))

        # get "bounds"
        left_bounds = target_lons - target_def.pixel_size_x / 2
        right_bounds = target_lons + target_def.pixel_size_x / 2
        top_bounds = target_lats + target_def.pixel_size_y / 2
        bottom_bounds = target_lats - target_def.pixel_size_y / 2

        lon_bounds = numpy.stack([left_bounds, right_bounds], axis=2)
        lat_bounds = numpy.stack([bottom_bounds, top_bounds], axis=2)

        lats = numpy.repeat(lat_bounds, 2, axis=2)
        lons = numpy.stack([lon_bounds, numpy.flip(lon_bounds, axis=2)], axis=2)
        lons.shape = (lon_bounds.shape[0], lon_bounds.shape[1], 4)

        # Pre-allocate memory for final result
        binned_file_data = None

        radius = self._file_def.get('INFO', {}).get('binRadius', 5e4)

        if not isinstance(self._file_data, (list, tuple)):
            self._file_data = [self._file_data]

        num_files = len(self._file_data)
        steps = num_files * (len(self._to_bin) * 2) + num_files
        step = 0
        filenum = 1

        t_start = time.time()
        for file in self._file_data:
            if file.sizes['x'] == 0 or file.sizes['y'] == 0:
                step += len(self._to_bin) * 2 + 1
                filenum += 1
                continue  # No data in this file to process.

            target_file = xarray.Dataset()

            # Figure the start and duration of this file
            # Don't assume sorted, though probably is
            file_timerange = file['datetime_start']
            start_datetime = file_timerange.min()
            stop_datetime = file_timerange.max()

            target_file.coords['datetime_start'] = start_datetime
            target_file.coords['datetime_length'] = stop_datetime - start_datetime

            if 'SO2_column_number_density_validity' in file:
                # Validity is not valid for gridded data.
                del file['SO2_column_number_density_validity']

            source_def = geometry.SwathDefinition(lons=file['longitude'],
                                                  lats = file['latitude'])

            resampler = None
            futures = []
            max_workers = None # Or None
            with ThreadPoolExecutor(max_workers = max_workers) as executor:
                for key in self._to_bin:
                    step += 1

                    if not key in file:
                        step += 1 # Add one for the "processing" step
                        continue

                    # target_file[key] = (file[key].dims, resampler.resample(file[key].data,
                        # fill_value=numpy.nan,
                        # ))
                    resample_data = file[key].data

                    # We need the data to be float type so we can have NaN values
                    if resample_data.dtype in (int, numpy.dtype('int32')):
                        resample_data = resample_data.astype(float)

                    future = executor.submit(run_resample, source_def, target_def,
                                             radius, resample_data)
                    futures.append((key, future))

                for key, future in futures:
                    step += 1

                    try:
                        target_file[key] = (file[key].dims, future.result())
                    except Exception as e:
                        logging.warning(f"Got error when binning {key}. Trying again...")
                        logging.debug(f"Error: {e}")

                        # hopefully we never need this, so only create it if we actually do
                        if resampler is None:
                            resampler = NumpyBilinearResampler(source_def, target_def, radius)

                        resample_data = file[key].data
                        if resample_data.dtype in (int, numpy.dtype('int32')):
                            resample_data = resample_data.astype(float)
                        target_file[key] = (
                            file[key].dims,
                            resampler.resample(resample_data, fill_value = numpy.nan)
                        )

            target_file.coords['latitude'] = (('y', 'x'), target_lats)
            target_file.coords['longitude'] = (('y', 'x'), target_lons)
            target_file.coords['latitude_bounds'] = (('y', 'x', 'corners'), lats)
            target_file.coords['longitude_bounds'] = (('y', 'x', 'corners'), lons)
            if binned_file_data is None:
                binned_file_data = target_file
            else:
                bfd_keys = set(binned_file_data.keys())
                tf_keys = set(target_file.keys())

                # Delete any variables that exist in the binend file data, but not the target file
                for key in bfd_keys - tf_keys:
                    del binned_file_data[key]

                # And the same for the reverse
                for key in tf_keys - bfd_keys:
                    del target_file[key]

                binned_file_data = xarray.concat([binned_file_data, target_file], 'file', compat = "no_conflicts")

            step += 1
            filenum += 1

        self._file_data = binned_file_data
        logging.debug(f"Completed binning in {time.time()-t_start} seconds")
        logging.debug("Completed %d steps (of %d)", step, steps)


def run_resample(source_def, target_def, radius, data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        resampler = NumpyBilinearResampler(source_def, target_def, radius)
        return resampler.resample(data, fill_value = numpy.nan)


def import_product(files, filters=None, options=None, fields = None):
    with NetCDFFile(files) as file:
        return file.import_data(filters, options, fields = fields)


if __name__ == "__main__": #pragma: nocover
    # Test code when calling this file directly. Feel free to modify to make
    # your own test cases
    START = time.time()
    FILE = "/Users/israel/Downloads/V2021362205348.SO2AI_JPSS-1.h5"
    # FILE = "/Users/israel/Desktop/Data/OMPS/2020-04-20/OMPS-NPP_NMSO2-PCA-L2_v1.1_2020m0420t111818_o00001_2020m0420t114132.h5"

    lat_from = 45
    _lat_to = 65
    lon_from = -180
    _lon_to = -140
    num_lat = round((_lat_to - lat_from) / .06)
    num_lon = round((_lon_to - lon_from) / .06)

    FILTER_STRING = ''

    data = import_product(FILE, FILTER_STRING)
    print(data['latitude_bounds'])
    print(data['longitude_bounds'])
#     with open("/tmp/test_data.pickle", 'rb') as file:
#         old_data = pickle.load(file)
    print("Ran in:", time.time() - START)
