#!/usr/bin/env python
# this is <VLPtools.py>
# ------------------------------------------------------------------------
#
# Copyright (c) 2023 by Thomas Forbriger (KIT, GPI, BFO)
#
# a module for handling VLP signal analysis
#
# REVISIONS and CHANGES
#    30/09/2023   V1.0   Thomas Forbriger
#
# ========================================================================
#
import modules.pazfilter as paz
from modules.invresponse import dumpresp
import numpy as np

def groundmotion(stin, inv, f0, quantity="displacement", verbose=True):
    """
    convert raw recordings to ground motion in a specific frequency band

    The two lowermost poles of the instrument are moved to f0.
    The signal values are converted to the specified quantity.

    parameters
    ----------
        stin : obspy.Stream
            three-component time series data
        inv : obspy.Inventory
            station metadata
        f0 : float
            frequency to which lowest poles of response are shifted / Hz
        quantity : str
            kinematic quantity to represent ground modtion
            may be:
            - acceleration
            - displacement
            - velocity
        verbose : bool
            be verbose

    returns
    -------
        obspy.Stream
            converted time series data
    """

    st=stin.copy()
    st.remove_sensitivity()

    if quantity == "displacement":
        st.integrate()
        units="displacement / $\mu$m"
        unitfac=1.e6
    elif quantity == "acceleration":
        st.differentiate()
        units="acceleration / nm s$^{-2}$"
        unitfac=1.e9
    elif quantity == "velocity":
        units="velocity / $\mu$m s$^{-1}$" #nm
        unitfac=1.e6
    else:
        print("ERROR: undefined quantity %s" % quantity)
        exit(3)

    pazsim=paz.PAZBWHP(1./f0,2)
    pazsim["zeros"]=[]

    for tr in st:
        if verbose:
            print("convert %s" % tr.id)
        tr.data *= unitfac
        tr.stats.units=units
        resp=inv.get_response(tr.id, datetime=tr.stats.starttime)
        if verbose:
            dumpresp(resp)
        seispaz=resp.get_paz()
        funits=0.
        if seispaz.pz_transfer_function_type == "LAPLACE (HERTZ)":
            funits=2.*np.pi
        elif seispaz.pz_transfer_function_type == "LAPLACE (RADIANS/SECOND)":
            funits=1.
        else:
            print("ERROR: unexpected type of response parameters!")
            exit()
        if verbose:
            print(seispaz)
        poles=paz.sortpz([(x*funits) for x in seispaz.poles])
        if verbose:
            print(poles)
            print(poles[0:2]) # Change to three if using lennarz seimometer
        paz.printpz(poles[0:2])
        pazseis={'zeros': [],
                 'poles': poles[0:2],
                 'gain': 1.0}
        pazfilt=paz.concatenate([pazsim,paz.reciprocal(pazseis)])
        if verbose:
            paz.printsys(pazfilt)
        tr.data=paz.sosfilter(tr.data, pazfilt, tr.stats.delta)

    return st

# ----- END OF VLPtools.py -----
