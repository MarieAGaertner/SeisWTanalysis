#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 15:56:47 2023

@author: Marie A. Gärtner
"""

__version__ = "1.0.0"

import logging
import configparser
import numpy as np
import pandas as pd
import xarray as xr

from obspy.clients.fdsn import Client
from obspy.core import UTCDateTime

import sys
import os
from scatseisnet import ScatteringNetwork
from modules.VLPtools import groundmotion

log = logging.getLogger(__name__)


class ScatTransform():
    """ Docu """

    def __init__(self, config):
        self._cfgfile = config
        self._parse_config()
        self._scattering_network()
        self._scattering_transform()

    def _parse_config(self):

        log.info("%s", self._cfgfile)
        config = configparser.ConfigParser()
        config.read(self._cfgfile)

        # Parameter scattering network
        self._segment_duration = config["seismogram"].getfloat("segment_duration")
        self._sampling_rate = config["seismogram"].getfloat("sampling_rate")
        self._taper_alpha = config["seismogram"].getfloat("taper_alpha")
        self._bank_parameter = (
            # first layer
            {"octaves": config["scatnet"].getint("octaves_layer1"),
             "resolution": config["scatnet"].getint("resolution_layer1"),
             "quality": config["scatnet"].getint("quality_layer1"),
             "normalize_wavelet": config["scatnet"]["normalize_wavelet"]},
            # second layer
            {"octaves": config["scatnet"].getint("octaves_layer2"),
             "resolution": config["scatnet"].getint("resolution_layer2"),
             "quality": config["scatnet"].getint("quality_layer2"),
             "normalize_wavelet": config["scatnet"]["normalize_wavelet"]},
        )

        self._pooling = config["scatnet"]["poolingtype"]

        # Paramerter seismogram
        self._seismogram_param = {
            "station": config["seismogram"]["station"],
            "starttime": config["seismogram"]["starttime"],
            "endtime": config["seismogram"]["endtime"],
            "network": config["seismogram"]["network"],
            "clientName": config["seismogram"]["clientName"],
            "overlap": config["seismogram"].getfloat("overlap"),
        }

        log.info("Sampling rate: %s", self._sampling_rate)

        # Directories
        self._directories = {
            "rootpath": config["directories"]["rootpath"],
            "scname": config["directories"]["sc_str"],
            }

        self._scdir = "%s%s/" % (
            self._directories['rootpath'],
            self._seismogram_param["station"],
        )

    def _scattering_network(self):
        log.info("Setup scattering network")
        self._samples_per_segment = int(self._segment_duration * self._sampling_rate)
        self._network = ScatteringNetwork(
            *self._bank_parameter,
            bins=self._samples_per_segment,
            sampling_rate=self._sampling_rate,
        )
        log.info("%s", self._network)

    def _scattering_transform(self):
        self._client = Client(self._seismogram_param["clientName"])
        starttime = UTCDateTime(self._seismogram_param["starttime"])
        endtime = UTCDateTime(self._seismogram_param["endtime"])
        dt_day = 24*60**2

        starttimes = np.arange(starttime, endtime, dt_day)
        endtimes = np.arange(starttime+dt_day, endtime+dt_day, dt_day)

        for t_s, t_e in zip(starttimes, endtimes):
            log.info(f"Processed day: {t_s}")
            self._t_start = t_s
            self._t_end = t_e

            if (self._seismogram_param["station"] == "IW08B") & ("2022-12-13" in str(t_s)):
                log.info("Skip day 2022-12-13 because of missing data.")
                continue

            self._load_seismogram()

            # Check for NaN and remove segment in case it contains any NaNs
            nan_indices = np.where(np.isnan(np.sum(np.array(self._segments), axis=(1, 2))))[0]

            if np.any(nan_indices):
                for nan_idx in nan_indices:
                    self._segments[nan_idx] = np.zeros_like(self._segments[nan_idx]) + 10

            log.info("Scattering transform")
            self._scattering_coefficients = self._network.transform(
                self._segments,
                taper_alpha=self._taper_alpha,
                reduce_type=eval(self._pooling)
            )

            if np.any(nan_indices):
                for nan_idx in nan_indices:
                    self._segments[nan_idx] = np.zeros_like(self._segments[nan_idx]) + np.nan

                    for sc in self._scattering_coefficients:
                        sc[nan_idx, ...] = np.zeros_like(sc[nan_idx, ...]) + np.nan
                        print(sc[nan_idx, ...])

            self._S0 = np.abs(np.array(self._segments))
            self._S0 = eval(f"lambda x: {self._pooling}(x, axis=-1)", {'np': np})(self._S0)

            self._save_scattering_coefficients()

        # Concatenate scattering coefficients
        self._concat_scat_coef()

    def _load_seismogram(self):
        # Load seismogram
        log.info("Load seimogram: %s - %s (See todo)", self._t_start, self._t_end)
        # To-do: If the recordings of one day have a gap, only the first part of the data is
        # processed. Change so the data after the gap is precessed aswell.
        self._st = self._client.get_waveforms(
            network=self._seismogram_param["network"],
            station=self._seismogram_param["station"],
            location="*",
            channel="*",
            starttime=self._t_start - self._segment_duration,
            endtime=self._t_end + self._segment_duration,
            attach_response=True,
        )
        self._inv = self._client.get_stations(
            network=self._seismogram_param["network"],
            station=self._seismogram_param["station"],
            starttime=self._t_start - self._segment_duration,
            endtime=self._t_end + self._segment_duration,
            level='response',
        )

        log.info("Seimogram raw: %s ", self._st)
        # Process seismogram
        log.info("Process seimogram")

        # Detrend
        self._st.detrend("linear")
        self._st.detrend("constant")

        # Remove response
        self._st = groundmotion(self._st, self._inv, f0=0.05, quantity="velocity", verbose=False)
        self._st.filter(type="lowpass", freq=45)

        # Resample
        self._st.resample(self._sampling_rate)

        # Merge streams in case of gaps or overlaps
        self._st.merge(method=1, fill_value=None)

        # Trim to time of interest
        self._st.trim(self._t_start, self._t_end)
        log.info("Seimogram before segmentation: %s ", self._st)

        # Chunk seismogram
        log.info("Chunk seimogram into segments of %s s.", self._segment_duration)
        self._segments = list()
        self._datetime = list()

        # Collect data and timestamps -- needs to be adapted for larger datasets
        for windowed_st in self._st.slide(
                window_length=self._segment_duration,
                step=self._segment_duration * self._seismogram_param["overlap"]):

            self._segments.append(np.array([tr.data[:-1] for tr in windowed_st]))

            if np.ma.is_masked(windowed_st[0].times("utcdatetime")[0]):
                if not self._datetime:
                    self._datetime.append(self._t_start)
                else:
                    self._datetime.append(
                        self._datetime[-1] +
                        self._segment_duration * self._seismogram_param["overlap"]
                    )
            else:
                self._datetime.append(windowed_st[0].times("utcdatetime")[0])

        log.info("Segment length: %s", len(self._segments))

        # Transform datetime array to pandas datetime format
        self._datetime = pd.to_datetime(self._datetime, utc=True, format='%Y-%m-%dT%H:%M:%S.%fZ')
        print(self._datetime)

    def _save_scattering_coefficients(self):
        log.info("Save scattering coefficients: %s - %s", self._t_start, self._t_end)
        channel = []
        for i in range(len(self._st)):
            channel.append(self._st[i].stats.channel)

        dataset = xr.Dataset(
            # data_vars = {'waveforms': (('calendartime','channel','time'), waveforms)},
            coords={
                'time': self._datetime.values,
                'channel': channel,
                'f1': self._network.banks[0].centers,
                'f2': self._network.banks[1].centers,
            },
            data_vars={
                "S0": (('time', 'channel'), self._S0),
                "S1": (('time', 'channel', 'f1'), self._scattering_coefficients[0]),
                "S2": (('time', 'channel', 'f1', 'f2'), self._scattering_coefficients[1]),
            },
            attrs={
                'station': self._seismogram_param["station"],
                'network': self._seismogram_param["network"],
                'sampling_rate': self._sampling_rate,
                'segment_duration': self._segment_duration,
            }
        )

        scpath = "%s%s_%s%s" % (
            self._scdir,
            self._directories['scname'][:-3],
            self._datetime[0].strftime("%Y%m%d"),
            self._directories['scname'][-3:]
        )
        os.makedirs(self._scdir, exist_ok=True)

        # Delete existing file
        if os.path.exists(scpath):
            os.remove(scpath)

        # Save scattering coeffitients
        dataset.to_netcdf(scpath)

    def _concat_scat_coef(self):
        log.info("Concatenate scattering coefficients and remove day files.")
        for filename in os.listdir(self._scdir):
            f = os.path.join(self._scdir, filename)
            if os.path.isfile(f) and ('scattering_coefficients_' in f):
                if 'combined_ds' not in locals():
                    combined_ds = xr.open_dataset(f)
                else:
                    ds = xr.open_dataset(f)
                    combined_ds = xr.concat([combined_ds, ds], dim="time")

                # remove file
                os.remove(f)

        combined_ds = combined_ds.sortby("time")
        combined_ds = combined_ds.drop_duplicates(dim="time")

        t0 = combined_ds['time'].values[0]
        t1 = combined_ds['time'].values[-1]
        dt = self._segment_duration * self._seismogram_param["overlap"]
        dt = pd.to_timedelta(dt, unit='s')
        time_complete = pd.date_range(t0, t1, freq=dt).to_numpy()

        ds = combined_ds.reindex({"time": time_complete})

        # save concatenated scattering coefitients
        scpath = os.path.join(self._scdir, self._directories['scname'])

        # Delete existing file
        if os.path.exists(scpath):
            os.remove(scpath)

        # Save scattering coeffitients
        ds.to_netcdf(scpath)
        ds.close()
