#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Paul Houssel
# License: MIT License

'''
This python program downloads and organise the dataset correctly in order
to be used by the Machine Learning scripts
'''

import pygit
from os import listdir

# Download the dataset
pygit.repos()
r = pygit.load("Mouse-Dynamics-Challenge")
r.fetch()
