import config

import argparse
import gc
import logging
import json
import math
import multiprocessing as mp
import os
import sys
import time
import warnings

from datetime import datetime
from io import BytesIO
from pathlib import Path

import numpy
import pytz
import pyproj
import pyqtgraph as pg
import xarray

from PIL import Image
from pycoast import ContourWriterAGG

from PySide2.QtGui import (QPainterPath,
                           QFont)

from PySide2.QtWidgets import (QApplication,
                               QWidget,
                               QVBoxLayout,
                               QHBoxLayout,
                               QLabel)


from PySide2.QtCore import (QSize,
                            QByteArray,
                            QBuffer,
                            QIODevice,
                            Qt)


from GradientScale import GradientWidget
from h5pyimport import import_product
from util import init_logging

DEBUG = False


def PolyArea(x, y):
    """Calculate the area of a polygon using native numpy math and the shoelace formula

    PARAMETERS
    ----------
    x : ndarray
        The x coordinates of the points that make up the polygon
    y : ndarray
        The y coordinates of the points that make up the polygon

    RETURNS
    -------
    area : float
        The calculated area of the polygon
    """
    S1 = numpy.sum(x * numpy.roll(y, -1, 1), 1)
    S2 = numpy.sum(y * numpy.roll(x, -1, 1), 1)

    area = .5 * numpy.absolute(S1 - S2)
    return area


def _generate_path(coord):
    path = QPainterPath()
    path.moveTo(*coord[0])
    path.lineTo(*coord[1])
    path.lineTo(*coord[2])
    path.lineTo(*coord[3])
    path.closeSubpath()
    return path


def _gen_sector_bounds(sectors):
    for sector in sectors:
        pix_size = float(sector['pixelSize'])  # In km
        center_lat = float(sector['centerLat'])
        center_lon = float(sector['centerLng'])
        width = float(sector['imageWidth'])
        height = float(sector['imageHeight'])

        half_width_meters = (width / 2) * 1000 * pix_size
        half_height_meters = (height / 2) * 1000 * pix_size

        # latitude is easy
        lat_degrees_per_meter = 1 / 111131.745
        lat_half_degrees = half_height_meters * lat_degrees_per_meter

        # Longitude is a tad more complicated, as it is based on latitude.
        # Use the center value for latitude.
        lon_degrees_per_meter = 1 / (math.cos(math.radians(center_lat)) * 111321)
        lon_half_degrees = half_width_meters * lon_degrees_per_meter

        # Add keys to match what we are expecting for values.
        # Simper than changing our expectations! :)
        sector['latFrom'] = center_lat - lat_half_degrees
        sector['latTo'] = center_lat + lat_half_degrees
        sector['longFrom'] = center_lon + lon_half_degrees
        sector['longTo'] = center_lon - lon_half_degrees
        sector['name'] = sector['sectorLabel']
        sector['showAllLabels'] = False


LAT_LON_PROJ = pyproj.Proj('epsg:4326', preserve_units=False)


def _initalize_image_widgets(file_date, band, dtype):
    """Set up the various QT widgets used to display the plot"""
    # Set up display widgets
    try:
        QApplication(sys.argv + ['-platform', 'offscreen'])  # So we can make widgets :)
    except RuntimeError as err:
        if "singleton" not in str(err):
            raise

    pg.setConfigOptions(background='#EEE', foreground='k')

    scale_font = QFont("Arial", 10)

    disp_widget = QWidget()
    disp_widget.setStyleSheet('background-color:rgba(0,255,0,255);padding:0px;margin:0px;')

    v_layout = QVBoxLayout()
    v_layout.setContentsMargins(0, 0, 0, 0)
    v_layout.setSpacing(0)

    view_box = pg.ViewBox(border={'width': 0},)

    plot_widget = pg.PlotWidget(disp_widget,
                                viewBox=view_box)

    scale_widget = GradientWidget()
    scale_widget.setOrientation("Horizontal")
    scale_widget.setFont(scale_font)
    scale_widget.setStyleSheet("background-color:white;")
    scale_widget.setFixedWidth(950)

    date_label = QLabel()
    if band is None:
        band = "Cloud"
    date_label.setText(f"{file_date.strftime('%Y-%m-%d %H:%M:%S')} UTC {dtype} {band}")
    date_label.setStyleSheet('color:#eee; background-color:rgba(0, 0, 0, 0.4); padding:2px 7px;')
    date_label_font = date_label.font()
    date_label_font.setPointSize(9)
    date_label.setFont(date_label_font)
    date_label.adjustSize()

    v_layout.addWidget(plot_widget)

    disp_widget.setLayout(v_layout)

    plot_item = plot_widget.getPlotItem()

    plot_item.hideAxis('left')
    plot_item.hideAxis('bottom')
    plot_item.hideButtons()

    img_width = 1000
    img_height = 800

    view_size = QSize(img_width, img_height)

    view_widget = plot_item.getViewWidget()
    view_widget.parent().setFixedSize(view_size)
    view_widget.adjustSize()

    return (plot_item, scale_widget, disp_widget, date_label)


class DataFile:
    use_spawn = True
    _data = None
    _du_val = None
    _normalized_du = None
    _bands = ('LowTrop', 'MidTrop')
    _request_url = 'https://volcview.wr.usgs.gov/vv-api/'
    _anc_request_url = 'https://avo-volcview.wr.usgs.gov/vv-api/'
    # _request_url = 'https://binarycascade.com/projects/vv-api/'
    _upload_path = 'imageApi/uploadImage'

    # To be set later, but defined here to keep linters happy
    _view_extents = None
    _proj_str = ''
    _laea_transformer = None

    def __init__(self, data_file, sectors = config.VOLCVIEW_SECTORS):
        # Check some values
        if not isinstance(sectors, (list, tuple, dict)):
            raise TypeError(f"img_sectors must be a list of sectors or a single "
                            f"sector dict, not {type(sectors)}")

        # If a single sector was passed in, make it into a list
        if isinstance(sectors, dict):
            sectors = (sectors, )

        # Sectors to generate images for
        self._sectors = sectors

        # File to generate images for
        self._file = data_file

        # Figure out the file type and date
        self._file_name = data_file.split('/')[-1]
        if self._file_name[:4] == "S5P_":
            self._heights = ['1km', '7km']
            self._data_type = 'TROPOMI'
            file_date_info = self._file_name.split("____")[1].split("_")[0]
            file_date = datetime.strptime(file_date_info, "%Y%m%dT%H%M%S")
        elif self._file_name[:4] == "OMPS":
            self._heights = ['TRL', 'TRM']
            self._data_type = 'OMPS'
            file_date_info = self._file_name.split("_")[3]
            file_date = datetime.strptime(file_date_info, "%Ym%m%dt%H%M%S")
        elif self._file_name.startswith('V'):
            self._heights = ['SO2index']
            self._data_type = "VIIRS"
            self._bands = ('SO2', )
            file_date_info = self._file_name[1:14]
            file_date = datetime.strptime(file_date_info, '%Y%j%H%M%S')

        self._file_date = file_date.replace(tzinfo =pytz.UTC)

        # Initalize some constants
        self._du_color_map = pg.ColorMap([0, .05, .1, .175, .25, .99, 1],
                                         [(255, 255, 255),
                                          (241, 187, 252),
                                          (53, 248, 244),
                                          (255, 225, 0),
                                          (248, 152, 6),
                                          (255, 19, 0),
                                          (255, 0, 0)])
        self._du_scale_labels = {0: "0 DU", .05: "1 DU", .1: "2 DU", .25: "5 DU", .6: "12 DU", 1: ">20 DU", }

        self._cloud_color_map = pg.ColorMap([0, 1], [(0, 0, 0), (255, 255, 255)])
        self._cloud_scale_labels = {0: "0%", 1: "100%", }

        self._mpctx = mp.get_context('spawn')

    def process_data(self):

        for idx, height in enumerate(self._heights):
            try:
                del self._data
            except AttributeError:
                pass
            else:
                gc.collect()

            if DEBUG:
                print("----------------------------------------")
            logging.info(f"Beginning data load for {self._file_name} {height}")

            try:
                self._load_data(height)
                if not self._data or not self._data['latitude'].any():
                    raise TypeError("Missing Data")
            except TypeError:
                logging.warning(f"No data found for {self._file_name}, {height}")
                continue

            # Process sectors in parallel, but only one period/height at a time so
            # we aren't trying to load too much data into memory
            self._run_processes(self._bands[idx])

            logging.info(f"Image generation for {height} complete.")

        logging.debug("Sector images generated. Generating Cloud image.")

        # We need to do a different load to calculate cloud cover
        # VIIRS doesn't have cloud data
        if self._data_type not in ("VIIRS"):
            try:
                self._load_data(validity=0, cloud_fraction='>=0')
                if not self._data or not self._data['latitude'].any():
                    raise TypeError("Missing data")
            except TypeError:
                logging.warning("No data found for cloud load")
            else:
                logging.debug("Cloud data loaded, begining processing")
                self._run_processes(gen_cloud=True)

    def _run_processes(self, *args, **kwargs):
        sector_processes = []
        if not DEBUG:
            if self.use_spawn:
                logging.debug("Using SPAWN to launch process")
                pool = self._mpctx.Pool(initializer = init_logging)
            else:
                logging.debug("Using normal FORK to launch process")
                pool = mp.Pool(initializer = init_logging)

        for sector in self._sectors:
            logging.debug("Launching generation process")
            if DEBUG:
                self._generate_sector(sector, *args, **kwargs)
            else:
                sector_result = pool.apply_async(self._generate_sector,
                                                 args=(sector, ) + args,
                                                 kwds = kwargs)

                sector_processes.append(sector_result)
                logging.debug("Process Status: %r", not sector_result.ready())
        if not DEBUG:
            pool.close()

        for idx, proc in enumerate(sector_processes):
            logging.debug("Waiting for process %i to complete", idx)
            for cnt in range(60):
                proc.wait(10)  # Wait for all sectors to be complete, but no more than two minutes
                if proc.ready():
                    break
                logging.debug("Still waiting for process after %i, status: %r", cnt, proc.ready())
            else:
                logging.warning("Image generation process failed to complete after 10 minutes. Will be terminated.")
        # All processes have either ended or timed-out. Terminate any remaining
        if not DEBUG:
            pool.terminate()

    def _load_data(self, height=None, validity=None, **kwargs):
        filters = [
            # These were some suggested "default" filters, but Taryn decided
            # she wanted volcview images to be more "raw"
            # "sensor_zenith_angle<62",
            # "solar_zenith_angle<70",
        ]

        if validity is not None:
            filters.append(f"SO2_column_number_density_validity>={validity}")
        else:
            filters.append("valid(SO2_column_number_density)",)

        for arg, value in kwargs.items():
            filters.append(f"{arg}{value}")

        filter_string = ";".join(filters)

        options = f'so2_column={height}' if height else ''
        try:
            self._data = import_product(self._file, filter_string, options)
        except Exception as e:
            logging.error(f"*****Got error when importing {height} product******")
            print(e)

    def _apply_filter(self, filter_, data_source = None):
        if data_source is None:
            data_source = self._data

        filter_ = xarray.DataArray(filter_, dims = ['time'])
        data = data_source.where(filter_, drop = True)
        return data

    def _generate_sector(self, sector, band=None, gen_cloud=False):
        logging.debug("Generation process launched for %s, generating cloud: %r", sector, gen_cloud)
        logging.info(f"Generating image for {sector['name']}")
        if self._data is None:
            logging.error("No data loaded for sector! Should never have gotten here!")
            return

        self._proj_str = f'+proj=laea +lat_0={sector["centerLat"]} +lon_0={sector["centerLng"]} +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs'

        self._laea_transformer = pyproj.Transformer.from_proj(LAT_LON_PROJ,
                                                              self._proj_str)

        lat_from, lat_to = (sector['latFrom'], sector['latTo'])
        lon_from, lon_to = (sector['longFrom'], sector['longTo'])

        # Make sure our longitude is in the "REAL" range
        if lon_from < -180:
            filt_lon_from = lon_from + 360
        elif lon_from > 180:
            filt_lon_from = lon_from - 360
        else:
            filt_lon_from = lon_from

        if lon_to > 180:
            filt_lon_to = lon_to - 360
        elif lon_to < -180:
            filt_lon_to = lon_to + 360
        else:
            filt_lon_to = lon_to

        # Just look at *this* sector
        # Start with a rough latitude/longitude filter
        # (with 1/2 degree latitude, 1 dgree longitude border)
        with numpy.errstate(invalid='ignore'):
            # Filter on density again so that any bins that wound up without data are removed.
            filter_items = [~numpy.isnan(self._data['SO2_column_number_density'])]

            filter_items.append(self._data['latitude'] >= (lat_from - .5))
            filter_items.append(self._data['latitude'] <= (lat_to + .5))

            logging.debug("Generating latitude filters")
            lat_filter = numpy.logical_and.reduce(filter_items)

            # Figure out longitude filter
            if filt_lon_to > filt_lon_from:
                filt_lon_from = [filt_lon_from, 180]
                filt_lon_to = [-180, filt_lon_to]
            else:
                filt_lon_from = [filt_lon_from]
                filt_lon_to = [filt_lon_to]

            lon_filters = []
            for start, stop in zip(filt_lon_from, filt_lon_to):
                lon_filters.append(numpy.logical_and(self._data['longitude'] <= (start + 1),
                                                     self._data['longitude'] >= (stop - 1)))

            logging.debug("Generating longitude filters")
            if len(lon_filters) > 1:
                lon_filter = numpy.logical_or(*lon_filters)
            else:
                lon_filter = lon_filters[0]

            logging.debug("Combining filters")
            post_filter = numpy.logical_and(lat_filter, lon_filter)

            # short-circuit filtering if no records would be retained
            if not post_filter.any():
                logging.info("No in-range data found for %s, %s",
                             band or "cloud", sector['sector'])
                return

            sector_data = self._apply_filter(post_filter)

        logging.debug("Rough filters applied")
        if not sector_data['latitude'].any():
            # No data for this set of parameters. Try the next
            logging.info("No data found for %s, %s", band or "cloud", sector['sector'])
            return

        # Figure out the bounds in laea projection
        pixel_bounds = numpy.stack((sector_data['latitude_bounds'],
                                    sector_data['longitude_bounds']),
                                   axis=-1)

        x_lat_lon = pixel_bounds[:, :, 1].reshape(pixel_bounds[:, :, 1].size)
        y_lat_lon = pixel_bounds[:, :, 0].reshape(pixel_bounds[:, :, 0].size)

        x_laea, y_laea = self._laea_transformer.transform(y_lat_lon, x_lat_lon,)

        x_laea = x_laea.reshape(int(x_laea.size / 4), 4)
        y_laea = y_laea.reshape(int(y_laea.size / 4), 4)

        # Add these to sector data so they get filtered along with
        # everything else
        sector_data['x_laea'] = (['time', 'corners'], x_laea)
        sector_data['y_laea'] = (['time', 'corners'], y_laea)

        # seperate the max/min x and y limits of each pixel so we can tell which
        # actually have area within the image
        x_max = numpy.nanmax(x_laea, axis = 1)
        x_min = numpy.nanmin(x_laea, axis = 1)
        y_max = numpy.nanmax(y_laea, axis = 1)
        y_min = numpy.nanmin(y_laea, axis = 1)

        meter_width = sector['pixelSize'] * 1000 * sector['imageWidth']  # km-->meters
        meter_height = sector['pixelSize'] * 1000 * sector['imageHeight']

        center_x, center_y = (0, 0)  # Always centered at 0, because that's how we defined our projection
        # Yes, this could be simplified, since our center is 0, but this keeps
        # flexability should that change, and explicitly spells out exactly what
        # we are doing here.
        x_range = [center_x - (meter_width / 2), center_x + (meter_width / 2)]
        y_range = [center_y - (meter_height / 2), center_y + (meter_height / 2)]

        # Second filter - now that we have translated the cordinate system,
        # trim down to *only* the area to be displayed.
        filters = [
            x_max > x_range[0],  # Right edge of pixel inside left edge of image
            x_min < x_range[1],  # Left edge of pixel inside right edge of image
            y_max > y_range[0],  # well, you get the idea
            y_min < y_range[1]
        ]

        final_filter = numpy.logical_and.reduce(filters)

        # Short-circuit filtering in border cases
        if not final_filter.any():
            logging.info("No in-range data found for %s, %s",
                         band, sector['sector'])
            return

        # Only filter if we are actually getting rid of something
        if not final_filter.all():
            sector_data = self._apply_filter(final_filter, sector_data)

        if not sector_data['latitude'].any():
            # No data for this set of parameters. Try the next
            logging.debug("No data found for %s, %s", band, sector['sector'])
            return

        logging.debug("Data filtered for sector succesfully")

        # Center point of pixels
        data_x, data_y = self._laea_transformer.transform(sector_data['latitude'],
                                                          sector_data['longitude'])
        laea_pixel_bounds = numpy.stack([sector_data['x_laea'],
                                         sector_data['y_laea']],
                                        axis=2)

        areas = PolyArea(sector_data['x_laea'],
                         sector_data['y_laea'])

        mass = areas * sector_data['SO2_column_number_density']  # in moles
        mass *= 64  # in grams
        total_mass = numpy.nansum(mass) * 1e-9  # Kilo Tonnes

        self._du_val = sector_data['SO2_column_number_density'] * 2241.15  # Conversion Factor from manual
        self._normalized_du = self._du_val * (1 / 20)
        self._normalized_du[self._normalized_du > 1] = 1
        self._normalized_du[self._normalized_du < 0] = 0

        _percentile_levels = (90, 95, 97, 99, 100)
        _percentiles = numpy.nanpercentile(self._du_val, _percentile_levels)
        _percentiles[_percentiles < 0] = 0
        _percentiles[_percentiles > 20] = 20

        _percentColors = self._du_color_map.map(_percentiles * (1 / 20),
                                                mode = 'qcolor')

        # When rounded to 5 digits, the color results are identical.
        # Doing the rounding significanly reduces the number of unique values,
        # therby enabling significant speed up by using a lookup table rather
        # than having to check each value individually.
        self._normalized_du = numpy.round(self._normalized_du, 5)

        # show_volc_names = sector.get('showAllLabels', True)
        # hide_all_names = sector.get('hideAllLabels', False)

        # Generate path objects for each pixel for graphing purposes.
        # To get the shape of each pixel, shift each one to 0,0 lower left bounding box
        shifted_coords = laea_pixel_bounds - numpy.min(laea_pixel_bounds, axis=1)[:, None, :]
        # We have to do min twice to get the single min value for each group of corner points
        # If we only did it once, X and Y would be scaled seperately, distorting the shape.
        scale_factors = numpy.max(numpy.max(shifted_coords, axis=1), axis=1)
        # Scale each pixel to fit within -0.5 - +0.5
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scaled_coords = (shifted_coords * (1 / scale_factors[:, None, None])) - .5
        # "Center" the scaled coordinates so the paths correctly represent the points
        scaled_coords -= (((numpy.max(scaled_coords, axis=1) -
                            numpy.min(scaled_coords, axis=1)) - 1) / 2)[:, None, :]

        pixel_paths = [_generate_path(x) for x in scaled_coords]

        # Have to keep disp_widget, otherwise python deletes things too soon.
        (plot_item, scale_widget,
         disp_widget, date_label) = _initalize_image_widgets(self._file_date,
                                                             band,
                                                             self._data_type)

        # Add the total mass to the date label
        if sector['sector'] == '1kmHIKI':
            date_label.setText(date_label.text() + f" {total_mass:.2f}kt")
            date_label.adjustSize()

        # Make color blocks to display the percentiles
        if not gen_cloud:
            _percentContainer = QWidget()
            _percentContainer.setObjectName("Percent Container")
            _percentContainer.setAutoFillBackground(False)
            _percentContainer.setStyleSheet('background-color:transparent')
            main_layout = QVBoxLayout(_percentContainer)
            main_layout.setObjectName("Main Layout")
            main_layout.setContentsMargins(0, 0, 0, 0)
            title = QLabel(_percentContainer)
            title.setText("Percentiles:")
            title.setAlignment(Qt.AlignLeft)
            title_font = title.font()
            title_font.setPointSize(8)
            title.setFont(title_font)
            main_layout.addWidget(title)
            percentLayout = QHBoxLayout()
            percentLayout.setObjectName("Percent Bar Layout")
            percentLayout.setContentsMargins(0, 0, 0, 0)
            percentLayout.setSpacing(0)
            main_layout.addLayout(percentLayout)

            for idx, color in enumerate(_percentColors):
                val = _percentiles[idx]
                widg = QWidget()
                lay = QVBoxLayout()
                lay.setObjectName(f"Val {idx} layout")
                lay.setContentsMargins(0, 0, 0, 0)
                widg.setLayout(lay)
                label = QLabel()
                label.setText(f"{_percentile_levels[idx]}<sup>th</sup><br>{str(round(val, 2))} DU")
                labelFont = label.font()
                labelFont.setPointSize(8)
                label.setFont(labelFont)
                label.setAlignment(Qt.AlignCenter)
                lay.addWidget(label)
                ss = f'background-color:{color.name()};border:1px solid black;'
                if idx != 0:
                    ss += "border-left:None;"

                widg.setStyleSheet(ss)
                percentLayout.addWidget(widg)

            _percentContainer.setGeometry(0, 0, 300, 18)
            _percentContainer.setLayout(main_layout)

        vbox = plot_item.getViewBox()
        vbox.disableAutoRange()

        vbox.setRange(xRange=x_range, yRange=y_range, padding=0)

        ###############################################################
        # Plot Generation Code
        ###############################################################
        def plot_dataset(dataset, color_map, scale_labels):
            # Only generate the brush once for each unique value
            lookup_table = {x: pg.mkBrush(color_map.map(x)) for x in numpy.unique(dataset)}
            brushes = [lookup_table[x] for x in dataset.data]

            scale_widget.setGradient(color_map.getGradient())
            scale_widget.setLabels(scale_labels)

            # Generate Plot
            plot = plot_item.plot(data_x, data_y,
                                  pen=None,
                                  symbolPen=None,
                                  symbolBrush=brushes,
                                  pxMode=False,
                                  symbolSize=scale_factors,
                                  symbol=pixel_paths)

            plot_item.getViewWidget().parent().grab()
            volcview_img = plot_item.getViewWidget().parent().grab()
            self._view_extents = vbox.viewRange()

            file_bytes = QByteArray()
            file_buffer = QBuffer(file_bytes)
            file_buffer.open(QIODevice.WriteOnly)
            volcview_img.save(file_buffer, "PNG")
            file_buffer.close()

            file_stream = BytesIO(file_bytes)
            pil_img = Image.open(file_stream)

            # find coverage percent(ish)
            width, height = pil_img.size
            total_count = width * height  # Should be 800,000, unless we changed the size of the images.

            # dump into a numpy array to count grey pixels
            as_array = numpy.array(pil_img)

            # the grey value we use is 238, so if all elements of axis 2 are 238,
            # then the pixel is grey.
            is_grey = numpy.all(as_array == 238, axis=2)
            # number that is False is non-grey, or covered, pixels
            # Not quite true due to scale bar, borders, etc.
            unique, counts = numpy.unique(is_grey, return_counts=True)
            non_grey = dict(zip(unique, counts))[False]

            covered_percent = non_grey / total_count

            # Don't send the image to volcview unless it has at least 15% coverage.
            # Allow 2% for the scale bar and other features.
            threshold = .17
            if sector['pixelSize'] == 5:
                threshold = .06

            if covered_percent > threshold:
                self._add_coastlines(pil_img)

                raw_data = QByteArray()
                buffer = QBuffer(raw_data)

                if not gen_cloud and not self._data_type in ('VIIRS'):
                    # "Save" the percentile bar to a bytes buffer, in PNG format
                    buffer.open(QIODevice.WriteOnly)
                    _percentContainer.grab().save(buffer, "PNG")
                    buffer.close()

                    # Use a bytes IO object to "read" the image into a PIL object
                    img_stream = BytesIO(raw_data)
                    with Image.open(img_stream) as img:
                        pil_img.paste(img,
                                      (5, 5),
                                      mask = img)

                # Add the scale bar and timestamp.
                scale_top = pil_img.height

                buffer.open(QIODevice.WriteOnly)
                scale_widget.grab()  # why? WHYYYYYYYY????
                scale_widget.grab().save(buffer, "PNG")
                buffer.close()

                img_stream = BytesIO(raw_data)
                with Image.open(img_stream) as img:
                    scale_top = pil_img.height - img.height - 10
                    pil_img.paste(img, (25, scale_top), mask = img)

                # Add the timestamp
                buffer.open(QIODevice.WriteOnly)
                date_label.grab().save(buffer, "PNG")
                buffer.close()

                img_stream = BytesIO(raw_data)
                with Image.open(img_stream) as img:
                    pil_img.paste(img,
                                  (pil_img.width - img.width - 51,
                                   scale_top - img.height - 5),
                                  mask = img)

                # Save an archive image
                logging.debug("Saving archive image for %s", band)
                filename = f"{self._file_date.strftime('%Y_%m_%d_%H%M%S')}-{band}-{self._data_type}.png"
                save_file = os.path.join(config.FILE_BASE, 'VolcView', sector['name'], filename)
                os.makedirs(os.path.dirname(save_file), exist_ok = True)
                pil_img.save(save_file, format = 'PNG')

            else:
                logging.info("Not enough coverage to bother with")

            plot_item.removeItem(plot)
        ###############################################################

        if gen_cloud:
            band = 'cloud'
            logging.debug("Plotting CLOUD dataset")
            plot_dataset(sector_data['cloud_fraction'], self._cloud_color_map,
                         self._cloud_scale_labels)
            logging.debug("CLOUD dataset plotted. function ends.")
        elif self._data_type == 'VIIRS':
            labels = {0: "0", 0.5: "SO2 Index", 1: "100"}
            plot_dataset(self._normalized_du, self._du_color_map, labels)
        else:
            plot_dataset(self._normalized_du, self._du_color_map, self._du_scale_labels)

    def _add_coastlines(self, img):
        x_range, y_range = self._view_extents
        # coerce into a py-coast format
        area_extent = (x_range[0], y_range[0], x_range[1], y_range[1])
        area_def = (self._proj_str, area_extent)

        # Get the path to the shape files
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        shape_dir = os.path.realpath(os.path.join(cur_dir, 'gshhg'))
        cw = ContourWriterAGG(shape_dir)
        cw.add_coastlines(img, area_def, resolution='h', width=1.0, level=1, outline=(200, 156, 45))


def main(data_file, use_spawn=True):
    """Load a data file and generate and upload VolcView
    images for any defined VolcView sectors covered by the data."""
    start = time.time()
    # Convert volcview sector definitions to our "native" format
    _gen_sector_bounds(config.VOLCVIEW_SECTORS)  # "converts" in-place.

    logging.info("Generating images")
    file_processor = DataFile(data_file)
    file_processor.use_spawn = use_spawn
    file_processor.process_data()
    logging.info("Completed run in %d seconds", time.time() - start)
    return


if __name__ == "__main__":
    init_logging()
    parser = argparse.ArgumentParser(description = "SO2 data file interface to VolcView")
    parser.add_argument("files", nargs = "*", default = [],
                        help = "SO2 data files to generate and upload VolcView images for")

    args = parser.parse_args()
    if not args.files:
        print("No files specified. Nothing to do.")
        exit(1)

    for arg in args.files:
        if os.path.isdir(arg):
            files = Path(arg).rglob('*.[hn][5c]')
        else:
            files = [arg]

        for file in files:
            main(str(file))

    exit(0)
