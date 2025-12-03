#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Marie Gärtner
Date: 17.11.2023
"""
import scattransform as stf
import logging

logging.basicConfig(format='%(asctime)s : %(message)s',  level=logging.INFO, force=True)
log = logging.getLogger(__name__)

log.info("This is scattransform version %s", stf.__version__)

config_file = "config.ini"

my_stf = stf.ScatTransform(config_file)


# stf_result = my_stf.get_result()