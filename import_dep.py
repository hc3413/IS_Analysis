# Standard library imports
import os
import datetime
import re
import math
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass
import readline

# Data manipulation and analysis
import numpy as np
import pandas as pd
import scipy
from scipy.interpolate import griddata
from scipy.optimize import fsolve
from scipy.stats import linregress

# Visualization - matplotlib and related
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm, ListedColormap, Normalize
from matplotlib import ticker
from matplotlib.figure import Figure
from matplotlib.ticker import LogFormatter, LogFormatterSciNotation
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib import cm
import scienceplots
from cycler import cycler
import seaborn as sns

# PowerPoint automation
from pptx import Presentation
from pptx.util import Inches, Pt, Cm
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from pptx.oxml import parse_xml
from pptx.enum.shapes import MSO_SHAPE









