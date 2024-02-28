import test
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

path = str(ROOT) + '\\'
file = 'val.txt'

with open(path + file) as f:
    content = f.readlines()

if __name__ == '__main__':
    test.prn(content)