#!/usr/bin/env python
# this is <seitoshio.py>
# ----------------------------------------------------------------------------
#
# Written in 2020 by Thomas Forbriger
# 
# read data in ascii format with essential header fields
# copied from readdata.py on 3.12.2020
#
# ----
# This program source code is licensed under a CC0 license.
# 
# To the extent possible under law, the author(s) have waived all copyright
# and related or neighboring rights to this source code. You can copy, modify,
# distribute and compile the code, even for commercial purposes, all without
# asking permission. 
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# 
# For the complete text of the license, please visit
# https://creativecommons.org/publicdomain/zero/1.0/
# ----
#
# 
# REVISIONS and CHANGES 
#    18/10/2017   V1.0   Thomas Forbriger
#    22/01/2018   V1.1   update code to work on python3 too
#    12/10/2018   V1.2   return a python dictionary
#    03/12/2020   V1.3   return an object of type 
#                        :class:`~obspy.core.trace.Trace`
# 
# ============================================================================
#
from obspy.core.trace import Trace
from obspy.core.trace import Stats
from obspy.core.utcdatetime import UTCDateTime
from obspy.core.util.attribdict import AttribDict
import sys
import re
import numpy as np
#
# ============================================================================
#
def extractheaderline(lines, keyword):
    """
    Extract the first header line matching the given keyword.
    This is a helper function for readascii.

    function parameters:
        :type lines:    list of strings
        :param lines:   lines read from data file including header lines
        :type keyword:  string
        :param keyword: header keyword

    returns:
        :rtype:     list of strings
        :return:    first header line matching given keyword
    """
    retval=list(filter((lambda x: re.match('^# %s:' % keyword,x)), 
        lines)).pop().strip()
    return retval

#
# ============================================================================
#
# read time series data from file
def readascii(name, verbose=False, demean=True):
  """
  Read a data file in Seitosh ascii format and properly handle file
  header. See the following location for a specification of the
  format:
  https://git.scc.kit.edu/Seitosh/Seitosh/-/tree/master/src/libs/libdatrwxx/ascii

  function parameters:
    :type name:      string
    :param name:     name of file to be read
    :type verbose:   bool, optional
    :param verbose:  switches on output verbosity if True
    :type demean:    bool, optional
    :param demean:   remove average of time series if True

  return value:
   :rtype: :class:`~obspy.core.trace.Trace`
   :return: trace with header data

  Attention: This function is not (yet) able to handle multi-track
  files. When reading a multi-track file, it will concatenate the
  samples of all tracks without any consistency check. Header fields
  are taken from the first track.
  """
  if verbose:
      print ("read data from file %s" % name)

# read all lines from data file
  thelines=open(name, encoding='utf-8').readlines()

# extract header parameters
  stats=Stats()

  dateline=extractheaderline(thelines, 'date')
  if len(dateline) > 0:
      stats.starttime=UTCDateTime(dateline[8:])

  dtline=extractheaderline(thelines, 'dt')
  if len(dtline) > 0:
      stats.delta=float(dtline[6:])
      stats.sampling_rate=1./stats.delta

  stationline=extractheaderline(thelines, 'station')
  if len(stationline) > 0:
      stats.station=stationline[11:]

  channelline=extractheaderline(thelines, 'channel')
  if len(channelline) > 0:
      stats.channel=channelline[11:]

  auxidline=extractheaderline(thelines, 'auxid')
  if len(auxidline) > 0:
      stats.network=auxidline[9:11]

# extract additional header data
  adict=AttribDict()
  adict["filename"]=name
  auxidline=extractheaderline(thelines, 'auxid')
  if len(auxidline) > 0:
      adict["auxid"]=auxidline[9:]
  instypeline=extractheaderline(thelines, 'instype')
  if len(instypeline) > 0:
      adict["instype"]=instypeline[11:]

  stats.update(adict)

# strip header lines from data and convert to an array of float values
  data=np.array(list(filter(lambda x: not re.match('^#', x), thelines)),'f')
# remove constant offset from data (if present)
  if demean:
      data=data-np.mean(data)

  stats.npts=len(data)
  trace=Trace(data, stats)

# report to user, if verbosity is selected
  if verbose:
      print ("%35s: %s"       % ("name of data file", name))
      print ("%35s: %s/%s/%s" % ("channel/station/network",
          stats.channel, stats.station, stats.network))
      print ("%35s: %s"       % ("time of first sample", stats.starttime))
      print ("%35s: %7.4f s"  % ("sampling interval", stats.delta))
      print ("%35s: %d"       % ("number of samples", stats.npts))
      print ("obspy header data:\n%s" % str(stats))

# return trace object
  return (trace)
#
# ----- END OF seitoshio.py ----- 
