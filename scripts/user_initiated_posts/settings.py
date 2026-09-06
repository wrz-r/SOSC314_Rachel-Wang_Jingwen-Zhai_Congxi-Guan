# -*- coding: utf-8 -*-

import json
import os


def env_int(name, default):
    value = os.getenv(name)
    if value is None or value == '':
        return default
    return int(value)


def env_list(name, default):
    value = os.getenv(name)
    if value is None or value == '':
        return default

    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = [item.strip() for item in value.split(',')]

    if isinstance(parsed, str):
        parsed = [parsed]

    return [item for item in parsed if item]


BOT_NAME = 'weibo'

SPIDER_MODULES = ['weibo.spiders']
NEWSPIDER_MODULE = 'weibo.spiders'


# Request settings

COOKIES_ENABLED = False
TELNETCONSOLE_ENABLED = False

DEFAULT_REQUEST_HEADERS = {
    'Accept':
    'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',

    'Accept-Language':
    'zh-CN,zh;q=0.9,en;q=0.8,en-US;q=0.7',

    'cookie': os.getenv('WEIBO_COOKIE', ''),
}


# Crawler settings

DOWNLOAD_DELAY = 10

LOG_LEVEL = 'INFO'



# Search parameters

KEYWORD_LIST = ['结婚']


# only crawl for original posts
WEIBO_TYPE = 1

# do not limit if photos are contained
CONTAIN_TYPE = 0

# national-wise search
REGION = ['全部']


# Date range

START_DATE = '2021-01-01'
END_DATE = '2021-12-31'


# Pagination

FURTHER_THRESHOLD = 46


# 0 = do not limit the amount of results
LIMIT_RESULT = 0


# IP / media settings

FETCH_IP = False

IP_REQUEST_TIMEOUT = 5

IMAGES_STORE = './'

FILES_STORE = './'


# Output

ITEM_PIPELINES = {
    'weibo.pipelines.DuplicatesPipeline': 300,
    'weibo.pipelines.CsvPipeline': 301,
}