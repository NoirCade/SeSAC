import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]

sys.path.append(str(ROOT))

import sysInfoTitle

if __name__ == '__main__':
    python_title_printer = sysInfoTitle.PythonTitlePrinter()
    python_title_printer.sysInfo()