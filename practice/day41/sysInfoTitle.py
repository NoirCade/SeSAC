import sys


class Line:
    def __init__(self,char='-',count=100):
        self.char = char
        self.count = count
        
    def LinePrn(self,char=None, count=None):
        if char is None:
            char = self.char
        if count is None:
            count = self.count
             
        print('\n▶', char * count)
        
class PythonTitlePrinter(Line):
    def sysInfo(self): 
        super().__init__()
        self.LinePrn() 
        print("▷ Python Version:", sys.version)
        print("▷ Python Implementation:", sys.implementation.name)
        print("▷ Python Compiler:", sys.version.split()[0])
        print("▷ Python Build Number:", sys.version.split()[2])
        self.LinePrn('#',100)


# if __name__ == '__main__':
    # 예제 사용
    # python_title_printer = PythonTitlePrinter()
    # python_title_printer.sysInfo()