import sys
from PyQt5.QtWidgets import QApplication, QWidget,QLabel
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

class MyApp(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('My First Application')
        self.move(300, 0)
        
        label2 = QLabel(self)
        pixmap = QPixmap('Ready.png')
        label2.setPixmap(pixmap)
        label2.setAlignment(Qt.AlignHCenter)
        self.resize(pixmap.width(),pixmap.height())
        self.show()

      

if __name__ == '__main__':
   app = QApplication(sys.argv)
#    print(len(sys.argv),sys.argv[0])
   ex = MyApp()
   sys.exit(app.exec_())
