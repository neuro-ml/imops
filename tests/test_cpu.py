import platform
import sys
from importlib.util import find_spec
from pathlib import Path

import pytest

from imops.src._utils import supports_x86_64_v3


X86_64_V3_FLAGS = {'avx', 'avx2', 'bmi1', 'bmi2', 'f16c', 'fma', 'abm', 'movbe'}


def test_x86_64_v3_detection():
    if platform.machine().lower() not in ('x86_64', 'amd64'):
        assert not supports_x86_64_v3()
        return

    cpuinfo = Path('/proc/cpuinfo')
    if not cpuinfo.exists():
        pytest.skip('CPU flags are known here only from /proc/cpuinfo')

    flags = set(next(line for line in cpuinfo.read_text().splitlines() if line.startswith('flags')).split())
    assert supports_x86_64_v3() == (X86_64_V3_FLAGS <= flags)


def test_kernels_follow_cpu():
    v3 = supports_x86_64_v3() and find_spec('imops.src._zoom_v3') is not None
    assert sys.modules['imops.src._zoom'].__name__ == ('imops.src._zoom_v3' if v3 else 'imops.src._zoom')
