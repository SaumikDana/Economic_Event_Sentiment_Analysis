"""
Common imports for workforce intelligence NLP projects
"""

# Standard library
import os
import re
import json
import time
import datetime
import traceback
from urllib.parse import urljoin

# Data processing
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt

# Web scraping and API
import requests
from bs4 import BeautifulSoup

# NLP and ML
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# XML parsing
import xml.etree.ElementTree as ET


import requests
from bs4 import BeautifulSoup
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import xml.etree.ElementTree as ET
from urllib.parse import urljoin