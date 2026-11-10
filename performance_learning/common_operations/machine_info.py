"""CPU identifier recorded on the benchmark host, not the report-rendering host."""
import pathlib
import platform
import re
import subprocess


def cpu_id(value):
    return re.sub(r'[^a-z0-9]+', '_', value.lower()).strip('_') or 'unknown_cpu'


def detect_cpu_id():
    model = ''
    if platform.system() == 'Linux':
        try:
            entries = pathlib.Path('/proc/cpuinfo').read_text().splitlines()
            for key in ('model name', 'Hardware', 'Processor'):
                model = next((line.split(':', 1)[1].strip() for line in entries
                              if ':' in line and line.split(':', 1)[0].strip() == key), '')
                if model:
                    break
        except OSError:
            pass
    elif platform.system() == 'Darwin':
        try:
            model = subprocess.check_output(
                ['sysctl', '-n', 'machdep.cpu.brand_string'], text=True,
                stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    return cpu_id(model or platform.processor() or ('unknown_' + platform.machine()))
