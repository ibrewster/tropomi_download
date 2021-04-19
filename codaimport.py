"""
codaimport.py - Functions for loading data from NetCDF files using the coda
    library
Copyright © 2020 Alaska Volcano Observatory
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
import sys
import threading
import time
import warnings

from functools import partial
from glob import glob

import coda
import numpy
import xarray

from pyresample import bilinear, geometry, create_area_def

if 'PySide2.QtWidgets' in sys.modules:
    from PySide2.QtCore import (QObject,
                                Signal)
else:
    logging.warning("PySide2 not found. Not reporting progress.")

    class QObject:
        """Dummy class so we can have a single definition of _Signaller"""

    class Signal:
        """
            Signal implementation that behaves somewhat simularly to the Qt
            signal/slot implemention, but without using Qt.
        """
        _callbacks = []
        _types = None

        def __init__(self, *args):
            """
                Initalize the signal with a list of expected arg types
            """
            self._types = args

        def connect(self, fcn):
            """
                "connect" a function to this signal to be called when this
                signal is emitted.
            """
            self._callbacks.append(fcn)

        def emit(self, *args):
            """
                "Emit" a signal by calling the associated callback function(s)
                with the provided args.
            """
            for callback in self._callbacks:
                callback(*args)


class _Signaller(QObject):
    progress = Signal(float)
    status = Signal(str, int, int)
    load_queue = None


# We need a QObject instance in order to be able to dispatch Qt Signals
signaller = _Signaller()

_FILE_DEFS = {}
_file_def = {}
_file_type = None

# Get a list of .py files from the file_formats directory adjacent to this file.
# These files contain definitions of how to import data from various file types
# into the format we expect.
# File definition files MUST contain a __TYPE__ variable, which is the first
# four characters of the file name for the files it handles, and a DEF
# dictionary containing the import parameters (fields, info, etc)
FORMAT_DIR = os.path.join(os.path.dirname(__file__),
                          'file_formats')
formats = [x[:-3] for x in os.listdir(FORMAT_DIR)
           if x.endswith('.py')]

# Load all file formats found in the formats directory
for fmt in formats:
    module = importlib.import_module(f'file_formats.{fmt}')
    _FILE_DEFS[module.__TYPE__] = module.DEF


_PROGRESS_QUEUE = None
_TERM_FLAG = multiprocessing.Event()


def _init_file_load_process(sig_queue, prog_queue, term_flag):
    """
        Initalize some global variables for use when multiprocessing.
        By being global, they can be populated when initallizing the process,
        and are available to all functions run in the process, including things
        like map() that can't take extra arguments.
    """
    # These items must be declared globally for multiprocessing purposes -
    # we can't just pass them to the function.
    # pylint: disable=global-statement
    global _PROGRESS_QUEUE
    global _TERM_FLAG

    _PROGRESS_QUEUE = prog_queue
    _TERM_FLAG = term_flag
    signaller.load_queue = sig_queue


def _load_file_data(file_def, filepath):
    coda.init()
    file_xa = xarray.Dataset()
    coda_file = coda.open(filepath)
    try:
        data_size = numpy.prod(coda.get_size(coda_file,
                                             file_def['fields']['latitude']['path']))
    except coda.CodacError:
        # Path does not exist in file. Bad file, apparently
        return {}

    # Generate the time array
    point_time_data = coda.fetch(coda_file,
                                 file_def['info']['point_time']['path'])
    point_time_data = point_time_data.flatten()

    operation = file_def['info']['point_time'].get('operation')
    if operation:
        point_time_data = operation(point_time_data)

    file_time_path = file_def['info'].get('file_time', {}).get('path', None)
    if file_time_path:
        file_time = coda.fetch(coda_file, file_time_path)
        point_time_data = point_time_data + file_time

    if point_time_data.size != data_size:
        num_repeat = data_size / point_time_data.size
        point_time_data = numpy.repeat(point_time_data, num_repeat)

    file_xa.coords['datetime_start'] = ("time", point_time_data)

    # Fetch desired data from file
    for field, spec in file_def['fields'].items():
        try:
            kill = signaller.load_queue.get_nowait()
        except (queue.Empty, AttributeError):
            pass  # Nothng to get
        else:
            if kill:
                _TERM_FLAG.set()

        if _TERM_FLAG.is_set():
            coda.close(coda_file)
            return {}

        path = spec['path']
        field_data = coda.fetch(coda_file, path)

        isinstance(field_data, numpy.ndarray)
        if 'operation' in spec:
            field_data = spec['operation'](field_data)

        flat_shape = (-1, *field_data.shape[file_def['info']['nDims']:])
        field_data.shape = flat_shape

        dim = ["time"]
        if len(flat_shape) > 1:
            dim.append('corners')

        if field in ['latitude', 'longitude',
                     'latitude_bounds', 'longitude_bounds']:
            file_xa.coords[field] = (dim, field_data)
        else:
            file_xa[field] = (dim, field_data)

        try:
            _PROGRESS_QUEUE.put('PROGRESS')
        except AttributeError:
            pass

    coda.close(coda_file)
    return file_xa


def _progress_thread(final):
    progress = 0
    while True:
        try:
            msg = _PROGRESS_QUEUE.get(timeout=.25)
        except queue.Empty:
            if _TERM_FLAG.is_set():
                logging.debug("Exiting progress thread due to cancel")
                break
        else:
            if msg == "QUIT":
                logging.debug("Exiting progress thread due to QUIT")
                break
            if msg == 'PROGRESS':
                progress += 1
                signaller.progress.emit((progress / final) * 100)
            if isinstance(msg, (list, tuple)) and msg[0] == 'STATUS':
                signaller.status.emit(msg[1], -1, 0)


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

            if section == 1:
                # Working on the operator. NON alpha numeric only
                if char in ['>', '<', '=']:
                    operator += char
                    continue

                section = 2
                # We are now on the value
                value += char
            elif section == 2:
                # remainder of the string is the value. Could short-circuit by using enum,
                # and slicing once we got here. Probably not worth the optimization though.
                value += char

        value = float(value)

        result.append((field, operator, value))
    return result


class _CodaFile:
    _filter_ops = {
        ">": op.gt,
        ">=": op.ge,
        "==": op.eq,
        "<=": op.le,
        "<": op.lt,
    }

    _file_data = None
    _file_idxs = []

    def import_product(self, files, filters=None, options=None):
        """
            Main function of this class, given a list of files, filters, and
            options, import the data from the specified file(s) into a
            pre-defined data structure

            Parameters
            ----------
            files : str or list
                If str, the file to import. If list, a list of files to import
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
        _TERM_FLAG.clear()
        signaller.load_queue = multiprocessing.Queue()

        global _PROGRESS_QUEUE
        _PROGRESS_QUEUE = multiprocessing.Queue()

        if self._file_data:
            del self._file_data
            self._file_data = None
            gc.collect()

        if not isinstance(files, (list, tuple)):
            files = [files]

        # Process wildcards in filenames
        _files = []
        for filedef in files:
            if "*" in filedef or "?" in filedef:
                _files += glob(filedef)
            else:
                _files.append(filedef)

        # Parse out the options
        if options is None:
            _options = {}
        elif not isinstance(options, str):
            raise TypeError(f"Options must be a string, not {type(options)}")
        else:
            _options = {x.split("=")[0]: x.split("=")[1] for x in options.split(';') if x}

        so2_column = _options.get('so2_column')

        # get the file def for this file type
        global _file_def
        global _file_type
        # We key off the first four characters of the file name. Scary, but file names are documented, so...
        file_name = _files[0].split('/')[-1]
        _file_type = file_name[:4]
        _file_def = _FILE_DEFS[_file_type]

        # Set the path to the proper SO2 product depending on filter entered (if any)
        if so2_column is None:
            so2_path = _file_def['fields']['SO2_column_number_density']['default_path']
        else:
            prefix = _file_def['fields']['SO2_column_number_density']['path_prefix']
            so2_path = f"{prefix}{so2_column}"

        _file_def['fields']['SO2_column_number_density']['path'] = so2_path

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

            if not filter_item[0] in tuple(_file_def['fields']) + ('datetime_start', ):
                raise ValueError(f"Cannot filter on non-existent variable {filter_item[0]}")

        # Concatenate data from all files into a single data structure
        coda.init()

        total_steps = len(_file_def['fields']) * len(_files) + len(_files) + len(filters)

        # From here on, we need to make sure this thread is told to quit at any point we
        # might exit this function, whether due to error or intent.
        progress_thread = threading.Thread(target=_progress_thread, args=(total_steps, ))
        progress_thread.start()

        try:
            # Estimate array size
            load_file_data = partial(_load_file_data, _file_def)
            num_files = len(_files)

            if num_files == 1:
                # No sense in doing multiprocessing if there is only one file to load.
                all_data = [load_file_data(_files[0])]
            else:
                process_count = 3
                # Use multiprocessing if we can, otherwise use a thread pool.
                # Thread pool is about 25% slower in testing, but still faster
                # than non-threaded.
                try:
                    pool = multiprocessing.Pool(process_count,
                                                initializer=_init_file_load_process,
                                                initargs=(signaller.load_queue,
                                                          _PROGRESS_QUEUE,
                                                          _TERM_FLAG))
                except AssertionError:
                    # Use threads (multiprocessing.dummy)
                    pool = mpd.Pool(process_count)

                all_data = pool.imap(load_file_data, _files)
                pool.close()

            self._file_idxs = []
            data_size = 0
            for file_data in all_data:
                if _TERM_FLAG.is_set():
                    try:
                        pool.join()  # wait for the processes to exit cleanly
                    except UnboundLocalError:
                        pass  # not using a pool

                    progress_thread.join()  # also make sure our progress thread is dead.
                    del _PROGRESS_QUEUE
                    del signaller.load_queue
                    logging.warning("Exiting load thread due to cancel")
                    return {}

                if not file_data:
                    continue  # no data from this file

                # Try to filter this file
                for filter_ in filters:
                    # Two types of filters: function filters like bin_spatial()
                    # and data filters like latitude<50
                    if isinstance(filter_, str):
                        func, args = filter_.split("(")
                        if func == 'bin_spatial':
                            break  # We can't bin until we have all the data, so we have to stop here.
                        # remove the closing parenthasis. Could probably do a chop
                        # as well.
                        args = args.replace(")", '')
                        args = args.split(",")
                        func = "_" + func

                        # Will throw an error if func is not defined
                        # better be a function that takes the proper arguments
                        getattr(self, func)(*args, file_data)
                    else:
                        file_data = self._apply_filter(*filter_, file_data)

                if not file_data or file_data['datetime_start'].size == 0:
                    continue  # no data from this file after filtering

                # Concatenate this file data with any existing data
                if self._file_data is None:
                    self._file_data = file_data
                else:
                    self._file_data = xarray.concat([self._file_data, file_data],
                                                    "time")

                data_size += file_data.sizes['time']
                self._file_idxs.append(data_size)

                _PROGRESS_QUEUE.put('PROGRESS')

            if self._file_data is None or 'latitude' not in self._file_data:
                _PROGRESS_QUEUE.put("QUIT")
                progress_thread.join()  # wait for exit
                return {}  # No data matching filters in these files

            signaller.status.emit("Applying Filters...", -1, 0)

            # We have our data loaded. Apply filters in order provided.
            # We need to apply again here so we can grid, if needed, and do
            # anything that comes after the gridding.
            for filter_ in filters:
                #  See if we need to stop
                try:
                    kill = signaller.load_queue.get_nowait()
                except queue.Empty:
                    pass  # Nothng to get
                else:
                    if kill is True:
                        _TERM_FLAG.set()

                # In case it was set elsewhere
                if _TERM_FLAG.is_set():
                    _PROGRESS_QUEUE.put("QUIT")
                    progress_thread.join()  # wait for exit
                    return {}

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

                _PROGRESS_QUEUE.put('PROGRESS')

            _PROGRESS_QUEUE.put("QUIT")
            progress_thread.join()  # wait for exit

            # In case it was set elsewhere
            if _TERM_FLAG.is_set():
                return {}
        except Exception as e:
            print("Caught exception:", e)
            logging.exception("Error when loading file:\n")
            # yes, we want to catch anything, because we need to make sure the
            # thread exits cleanly. No exceptions. Even something like a keyboard interupt
            _PROGRESS_QUEUE.put("QUIT")
            # set the term flag as well, just to make sure the message gets through.
            # Probably overkill.
            _TERM_FLAG.set()
            progress_thread.join()  # wait for exit
            raise

        return self._file_data

    def _update_file_idxs(self, data_filter):
        new_idxs = []
        for idx, stop in enumerate(self._file_idxs):
            if idx == 0:
                start = 0
                new_start = 0
            else:
                start = self._file_idxs[idx - 1]
                new_start = new_idxs[idx - 1]

            point_count = numpy.count_nonzero(data_filter[start:stop])
            new_stop = new_start + point_count
            new_idxs.append(new_stop)

        # See if we start with zero
        while new_idxs and new_idxs[0] == 0:
            # Empty first file
            del new_idxs[0]

        # Eliminate duplicates by passing through a dictionary
        self._file_idxs = list(dict.fromkeys(new_idxs))

    def _apply_filter(self, field, operator, value, data=None):
        if data is None:
            data = self._file_data

        _data_filter = self._filter_ops[operator](data[field], value)

        if _data_filter.all():
            return data  # Not removing any data, no need to go further.

        # Adjust file stops
        if data is self._file_data:
            self._update_file_idxs(_data_filter)
            self._file_data = data = data.where(_data_filter, drop = True)
        else:
            data = data.where(_data_filter, drop = True)

        return data

    def _valid(self, key, data=None):
        set_self = False
        if data is None:
            data = self._file_data
            set_self = True

        _data_filter = ~numpy.isnan(data[key])

        if _data_filter.all():
            return  # Not removing any data, no need to go further.

        if data is self._file_data:
            self._update_file_idxs(_data_filter)

        data = data.where(_data_filter, drop = True)
        if set_self:
            self._file_data = data

        return data

    def _longitude_range(self, lon_from, lon_to, data=None):
        """Always east to west lat from to lat to"""
        set_self = False
        if data is None:
            data = self._file_data
            set_self = True

        lon_from = float(lon_from)
        lon_to = float(lon_to)

        if lon_from < -180:
            lon_from = (lon_from + 360)

        if lon_to > 180:
            lon_to = -1 * (360 - lon_to)

        if lon_from > lon_to:
            lon_from = [lon_from, -180]
            lon_to = [180, lon_to]
        else:
            lon_from = [lon_from]
            lon_to = [lon_to]

        filters = []
        longitude_data = data['longitude']
        for start, stop in zip(lon_from, lon_to):
            filters.append(numpy.logical_and(longitude_data >= start, longitude_data <= stop))

        if len(filters) > 1:
            _data_filter = numpy.logical_or(*filters)
        else:
            _data_filter = filters[0]

        if _data_filter.all():
            return  # Not removing any data, no need to go further.

        if data is self._file_data:
            self._update_file_idxs(_data_filter)

        data = data.where(_data_filter, drop = True)
        if set_self:
            self._file_data = data

        return data

    def _bin_spatial(self, num_lat, lat_from, lat_step, num_lon, lon_from, lon_step,
                     proj=None):
        # When loading file, keep list of indexes for each data file. Then bin in chunks, with each
        # chunk becoming a layer in the result data set.

        if 'SO2_column_number_density_validity' in self._file_data:
            # Validity is not valid for gridded data.
            del self._file_data['SO2_column_number_density_validity']

        # Make sure everything is correctly typed
        num_lat = int(num_lat) - 1
        lat_from = float(lat_from)
        lat_step = float(lat_step)
        lat_to = lat_from + (num_lat * lat_step)

        num_lon = int(num_lon) - 1
        lon_from = float(lon_from)
        lon_step = float(lon_step)
        lon_to = lon_from + (num_lon * lon_step)

        if lon_to > 180:
            # Shift 360 degrees. With my changes, pyresample can cross the dateline to the
            # negitive, but not to the positive.
            lon_from -= 360
            lon_to -= 360

        # try gridding the data
        area_extent = (lon_from, lat_from,
                       lon_to, lat_to)

        if isinstance(proj, str):
            proj = pickle.loads(bytes.fromhex(proj))

        proj_dict = proj if proj is not None else {'proj': 'lonlat', 'over': True, 'preserve_units': False, }

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
#         num_files = len(self._file_idxs)
#         binned_size = target_def.shape

#         # Pre-allocate the memory for the binned data to live in.
#         for key in self._file_data:
#             if (key in _file_def['fields'] and _file_def['fields'][key]['bin']) or key in ['latitude', 'longitude']:
#                 binned_file_data[key] = numpy.zeros((num_files, *binned_size))
#             elif key in ['latitude_bounds', 'longitude_bounds']:
#                 binned_file_data[key] = numpy.zeros((num_files, *binned_size, 4))
#             else:
#                 binned_file_data[key] = numpy.zeros((num_files, ))
#
#         # Add the datetime_length field for binned data
#         binned_file_data['datetime_length'] = numpy.zeros((num_files, ))

        for file_idx, stop in enumerate(self._file_idxs):
            try:
                kill = signaller.load_queue.get_nowait()
            except queue.Empty:
                pass  # Nothng to get
            else:
                if kill:
                    _TERM_FLAG.set()

            if _TERM_FLAG.is_set():
                return

            signaller.status.emit(f"Stacking Data ({file_idx}/{len(self._file_idxs)})...",
                                  file_idx, len(self._file_idxs))

            if file_idx == 0:
                start = 0
            else:
                start = self._file_idxs[file_idx - 1]

            binned_data = self._bin_file((start, stop), target_def)

            binned_data.coords['latitude'] = (('x', 'y'), target_lats)
            binned_data.coords['longitude'] = (('x', 'y'), target_lons)
            binned_data.coords['latitude_bounds'] = (('x', 'y', 'corners'), lats)
            binned_data.coords['longitude_bounds'] = (('x', 'y', 'corners'), lons)

            if binned_file_data is None:
                binned_file_data = binned_data
            else:
                binned_file_data = xarray.concat([binned_file_data, binned_data], 'file')

        self._file_data = binned_file_data

    def _bin_file(self, src_range, target_def):
        source_def = geometry.SwathDefinition(lons=self._file_data['longitude'][src_range[0]:src_range[1]],
                                              lats=self._file_data['latitude'][src_range[0]:src_range[1]])

        nan_slice = False

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                radius = _file_def.get('info', {}).get('binRadius', 5e4)
                t_params, s_params, input_idxs, idx_ref = bilinear.get_bil_info(source_def, target_def, radius=radius)
            # valid_input_index, valid_output_index, index_array, distance_array = \
            #    kd_tree.get_neighbour_info(source_def, target_def, 5000, neighbours=1)
        except (IndexError, ValueError):
            # No data from this file falls within the slice
            nan_slice = True

        # create target file data
        target_file = xarray.Dataset()

        # rebin any data needing rebinned.
        for key in [key for key in self._file_data if key in _file_def['fields'] and _file_def['fields'][key]['bin']]:
            if nan_slice:
                binned_data = numpy.full(target_def.shape, numpy.nan,
                                         dtype=float,)
            else:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    key_data = self._file_data[key][src_range[0]:src_range[1]].data
                    binned_data = bilinear.get_sample_from_bil_info(key_data,
                                                                    t_params, s_params, input_idxs, idx_ref,
                                                                    output_shape=target_def.shape)
            dim = ('x', 'y')
            if len(binned_data.shape) == 3:
                dim += ('corners')
            target_file[key] = (dim, binned_data)
#            target_file[key] = binned_data

        # Figure the start and duration of this file
        # Don't assume sorted, though probably is
        file_timerange = self._file_data['datetime_start'][src_range[0]:src_range[1]]
        start_datetime = file_timerange.min()
        stop_datetime = file_timerange.max()

        target_file.coords['datetime_start'] = start_datetime
        target_file.coords['datetime_length'] = stop_datetime - start_datetime

        return target_file


_coda_file_instance = _CodaFile()

import_product = _coda_file_instance.import_product

if __name__ == "__main__":
    # Test code when calling this file directly. Feel free to modify to make
    # your own test cases
    from . import wingdbstub
    START = time.time()
    FILE = '/tmp/Transfer/TROPOMIData/NRTI/2021-04-15/S5P_NRTI_L2__SO2____20210415T135653_20210415T140153_18162_01_020104_20210415T154113.download'

    lat_from = 45
    _lat_to = 65
    lon_from = -180
    _lon_to = -140
    num_lat = round((_lat_to - lat_from) / .06)
    num_lon = round((_lon_to - lon_from) / .06)

    FILTER_STRING = ''

    data = import_product(FILE, FILTER_STRING)
    print(data)
#     with open("/tmp/test_data.pickle", 'wb') as file:
#         pickle.dump(data, file)
    print("Ran in:", time.time() - START)
