import sys

PY2 = sys.version_info[0] == 2
PY3 = sys.version_info[0] == 3

if PY3:
    STRING_TYPES = str
else:
    STRING_TYPES = basestring
