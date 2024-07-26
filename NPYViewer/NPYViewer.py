import os.path
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import *
from pathlib import Path
import numpy as np
from numpy import asarray
from numpy import savetxt
import pandas as pd
import csv
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from scipy.io import savemat
import networkx as nx


version="1.28"

def isint(s):
    try:
        print(int(s))
        return True
    except ValueError:
        return False


def isfloat(s):
    try:
        print(float(s))
        return True
    except ValueError:
        return False



def openNPY_CLI_noGUI (filename):

        if ".npy" in filename:
            data = np.load(filename, allow_pickle=True)
        else:
            data = np.array(pd.read_csv(filename).values.tolist())

        npyfile = NPYfile(data, filename)
        print(npyfile)
        print(data)


class NPYfile():
    def __init__(self, data, filename):
        self.data = data
        self.filename = filename

    def __str__(self):
        if hasattr(self.data, 'dtype'):
            return "Filename = " + str(self.filename) + " \nDtype = " + str(self.data.dtype) + "\nShape = " + str(
                self.data.shape)
        else:
            return "Filename = " + str(self.filename) + " \nDtype = " + "\nShape = " + str(self.data.shape)



            
class MainApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.left = 0
        self.top = 0
        self.width = 800
        self.height = 640
        # np.save('/home/user/tst.npy', np.array([255.0,50.0,10.0,5.0]))
        self.npyfile = None
        self.initUI()

    def saveAs(self):
        home = str(Path.home())
        if self.npyfile is not None:
            home = os.path.dirname(self.npyfile.filename)
        path = QFileDialog.getSaveFileName(
            self, 'Save File', home, 'NPY (*.npy);;CSV(*.csv);;MAT(*.mat)')[0]
        # path = QFileDialog.getSaveFileName(
        #    self, 'Save File', home, 'CSV(*.csv)')[0]
        if path != "" and ".csv" in path:
            with open((path.replace(".csv", "") + ".csv"), 'w') as stream:
                writer = csv.writer(stream)
                for row in range(self.tableWidget.rowCount()):
                    rowdata = []
                    for column in range(self.tableWidget.columnCount()):
                        item = self.tableWidget.item(row, column)
                        if item is not None:
                            rowdata.append(item.text())

                        else:
                            rowdata.append('')
                    writer.writerow(rowdata)
        else:
            OutMatrix = []
            for row in range(self.tableWidget.rowCount()):
                rowdata = []
                for column in range(self.tableWidget.columnCount()):
                    item = self.tableWidget.item(row, column)
                    if item is not None:
                        #if item.text().isnumeric():
                        rowdata.append(float(item.text()))

                if rowdata != []:
                    OutMatrix.append(rowdata)
            OutMatrix = np.array(OutMatrix)
            if ".csv" in path:
                np.save(path, np.array(OutMatrix))
            if ".mat" in path:
                mdic = {"ans": OutMatrix}
                print(OutMatrix)
                savemat(path, mdic)

    def openNPY(self):
        if os.path.exists("lastpath.npy"):
            home = str(np.load("lastpath.npy"))
            print(home)
        else:
            home = str(Path.home())
        if self.npyfile is not None:
            home = os.path.dirname(self.npyfile.filename)
        filename = QFileDialog.getOpenFileName(self, 'Open .NPY file', home, ".NPY files (*.npy);;.CSV files (*.csv)")[
            0]
        if filename != "":
            if ".npy" in filename:
                data = np.load(filename, allow_pickle=True)
            else:
                data = np.array(pd.read_csv(filename).values.tolist())

            npyfile = NPYfile(data, filename)
            print(npyfile)
            self.setWindowTitle("NPYViewer v." +version+": "+ npyfile.filename)
            self.infoLb.setText("NPY Properties:\n" + str(npyfile))
            self.tableWidget.clear()

            # initialise table
            self.tableWidget.setRowCount(data.shape[0])
            dtype_dim = len(npyfile.data.dtype)  # 0, if plain dtype, 1 or bigger if compound dtype
            if data.ndim > 1:
                self.tableWidget.setColumnCount(data.shape[1])
            elif dtype_dim > 0:
                self.tableWidget.setColumnCount(dtype_dim)
            else:
                self.tableWidget.setColumnCount(1)

            # fill data
            if data.ndim > 1:
                for i, value1 in enumerate(npyfile.data):  # loop over items in first column
                    for j, value in enumerate(value1):
                        self.tableWidget.setItem(i, j, QTableWidgetItem(str(value)))
            elif dtype_dim > 0:
                for i, value1 in enumerate(npyfile.data):
                    for j, col_name in enumerate(npyfile.data.dtype.names):
                        self.tableWidget.setItem(i, j, QTableWidgetItem(str(value1[col_name])))
            else:
                for i, value1 in enumerate(npyfile.data):  # loop over items in first column
                    self.tableWidget.setItem(i, 0, QTableWidgetItem(str(value1)))

            self.npyfile = npyfile
            path = os.path.dirname(filename)
            np.save("lastpath.npy", path)


    def openNPY_CLI(self,filename):

            if ".npy" in filename:
                data = np.load(filename, allow_pickle=True)
            else:
                data = np.array(pd.read_csv(filename).values.tolist())

            npyfile = NPYfile(data, filename)
            print(npyfile)
            self.setWindowTitle("NPYViewer v." +version+": " +npyfile.filename)
            self.infoLb.setText("NPY Properties:\n" + str(npyfile))
            self.tableWidget.clear()

            # initialise table
            self.tableWidget.setRowCount(data.shape[0])
            dtype_dim = len(npyfile.data.dtype)  # 0, if plain dtype, 1 or bigger if compound dtype
            if data.ndim > 1:
                self.tableWidget.setColumnCount(data.shape[1])
            elif dtype_dim > 0:
                self.tableWidget.setColumnCount(dtype_dim)
            else:
                self.tableWidget.setColumnCount(1)

            # fill data
            if data.ndim > 1:
                for i, value1 in enumerate(npyfile.data):  # loop over items in first column
                    for j, value in enumerate(value1):
                        self.tableWidget.setItem(i, j, QTableWidgetItem(str(value)))
            elif dtype_dim > 0:
                for i, value1 in enumerate(npyfile.data):
                    for j, col_name in enumerate(npyfile.data.dtype.names):
                        self.tableWidget.setItem(i, j, QTableWidgetItem(str(value1[col_name])))
            else:
                for i, value1 in enumerate(npyfile.data):  # loop over items in first column
                    self.tableWidget.setItem(i, 0, QTableWidgetItem(str(value1)))

            self.npyfile = npyfile
            path = os.path.dirname(filename)
            np.save("lastpath.npy", path)


            
    def createMenu(self):

        exitAct = QAction(QIcon('exit.png'), '&Exit', self)
        exitAct.setShortcut('Ctrl+Q')
        exitAct.setStatusTip('Exit application')
        exitAct.triggered.connect(qApp.quit)

        openAct = QAction(QIcon('Open.png'), '&Open', self)
        openAct.setShortcut('Ctrl+O')
        openAct.setStatusTip('Open .NPY file')
        openAct.triggered.connect(self.openNPY)
        self.statusBar()

        saveAct = QAction(QIcon('Save.png'), '&Save As', self)
        saveAct.setShortcut('Ctrl+S')
        saveAct.setStatusTip('Save As')
        saveAct.triggered.connect(self.saveAs)
        self.statusBar()

        grayscalevVewAct = QAction(QIcon(None), '&View as Grayscale Image', self)
        grayscalevVewAct.setShortcut('Ctrl+V')
        grayscalevVewAct.setStatusTip('View as Grayscale')
        grayscalevVewAct.triggered.connect(self.grayscaleView)
        self.statusBar()

        View3dAct = QAction(QIcon(None), 'View &3D Point Cloud', self)
        View3dAct.setShortcut('Ctrl+3')
        View3dAct.setStatusTip('View 3D Point Cloud')
        View3dAct.triggered.connect(self.View3dPoints)
        self.statusBar()

        View3dImgAct = QAction(QIcon(None), 'View as &HeightMap', self)
        View3dImgAct.setShortcut('Ctrl+H')
        View3dImgAct.setStatusTip('View as HeightMap')
        View3dImgAct.triggered.connect(self.ViewImageHeightMap)
        self.statusBar()

        ViewTimeSeriesAct= QAction(QIcon(None), 'View as &Time Series', self)
        ViewTimeSeriesAct.setShortcut('Ctrl+S')
        ViewTimeSeriesAct.setStatusTip('View as TimeSeries')
        ViewTimeSeriesAct.triggered.connect(self.ViewTimeseries)
        self.statusBar()

        ViewGraphSeriesAct= QAction(QIcon(None), 'View as Directional &Graph', self)
        ViewGraphSeriesAct.setShortcut('Ctrl+G')
        ViewGraphSeriesAct.setStatusTip('View as Graph')
        ViewGraphSeriesAct.triggered.connect(self.ViewGraphSeriesAct)
        self.statusBar()
        
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&Functionalities')
        fileMenu.addAction(openAct)
        fileMenu.addAction(saveAct)
        fileMenu.addAction(grayscalevVewAct)
        fileMenu.addAction(View3dAct)
        fileMenu.addAction(View3dImgAct)
        fileMenu.addAction(ViewTimeSeriesAct)
        fileMenu.addAction(ViewGraphSeriesAct)
        fileMenu.addAction(exitAct)

    def grayscaleView(self):
        OutMatrix = []
        for row in range(self.tableWidget.rowCount()):
            rowdata = []
            for column in range(self.tableWidget.columnCount()):
                item = self.tableWidget.item(row, column)
                # print(item.text())
                if item is not None:
                    rowdata.append(np.float32(item.text()))

            if len(rowdata) > 0 and rowdata != None:
                OutMatrix.append(rowdata)

        OutMatrix = np.array(OutMatrix)
        plt.imshow(OutMatrix, cmap='gray')
        plt.show()
        return

    def ViewGraphSeriesAct(self):
        OutMatrix = []
        for row in range(self.tableWidget.rowCount()):
            rowdata = []
            for column in range(self.tableWidget.columnCount()):
                item = self.tableWidget.item(row, column)
                # print(item.text())
                if item is not None:
                    rowdata.append(np.float32(item.text()))

            if len(rowdata) > 0 and rowdata != None:
                OutMatrix.append(rowdata)

        OutMatrix = np.array(OutMatrix)


        G = nx.DiGraph(OutMatrix)

        # Set the position of the nodes in the graph
        pos = nx.spring_layout(G)

        # Draw the graph with labels and edges
        labels = {node: str(int(node)+1) for node in G.nodes()}

        nx.draw_networkx_labels(G, pos,labels=labels)
        nx.draw_networkx_edges(G, pos)

        nx.draw_networkx_edge_labels(G, pos, edge_labels={(i, j): OutMatrix[i][j] for i, j in G.edges()})

        # Draw the nodes with the same color
        nx.draw_networkx_nodes(G, pos, node_color='b')


        # Set the title and show the plot
        plt.title("Directional Graph")
        plt.show()

    def ViewImageHeightMap(self):
        OutMatrix = []
        for row in range(self.tableWidget.rowCount()):
            rowdata = []
            for column in range(self.tableWidget.columnCount()):
                item = self.tableWidget.item(row, column)

                if item is not None:
                    if item.text():
                        rowdata.append(np.float32(item.text()))
            if len(rowdata) > 0 and rowdata != None:
                OutMatrix.append(rowdata)
        # print(OutMatrix)
        HeightMap = []
        for x, row in enumerate(OutMatrix):
            for y, val in enumerate(row):
                HeightMap.append([x, y, val])
        OutMatrix = np.array(HeightMap)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        xs = OutMatrix[:, 0]

        ys = OutMatrix[:, 1]
        zs = OutMatrix[:, 2]
        ax.plot_trisurf(xs, ys, zs,
                        cmap='Greys_r', edgecolor='none');

        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')

        plt.show()
        return

    def View3dPoints(self):
        OutMatrix = []
        for row in range(self.tableWidget.rowCount()):
            rowdata = []
            for column in range(self.tableWidget.columnCount()):
                item = self.tableWidget.item(row, column)

                if item is not None:
                    if item.text():
                        rowdata.append(np.float32(item.text()))
            if len(rowdata) > 0 and rowdata != None:
                OutMatrix.append(rowdata)
        # print(OutMatrix)
        OutMatrix = np.array(OutMatrix)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        xs = OutMatrix[:, 0]

        ys = OutMatrix[:, 1]
        zs = OutMatrix[:, 2]
        ax.scatter(xs, ys, zs, c='r', marker='o')

        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.set_zlabel('Z Axis')

        plt.show()
        return
    
    def ViewTimeseries(self):
        OutMatrix = []
        for row in range(self.tableWidget.rowCount()):
            rowdata = []
            for column in range(self.tableWidget.columnCount()):
                item = self.tableWidget.item(row, column)

                if item is not None:
                    if item.text():
                        rowdata.append(np.float32(item.text()))
            if len(rowdata) > 0 and rowdata != None:
                OutMatrix.append(rowdata)
        # print(OutMatrix)
        #OutMatrix = np.array(OutMatrix)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        indices=range(0,len(OutMatrix))
        plt.plot(indices, OutMatrix, label='values', linewidth=3)

        ax.set_xlabel('Time Unit')
        ax.set_ylabel('Values')

        plt.show()
        return

    
    def initUI(self):
        self.createMenu()

        self.infoLb = QLabel("NPY Properties:")
        self.tableWidget = QTableWidget()
        self.tableWidget.setRowCount(100)
        self.tableWidget.setColumnCount(100)

        # table selection change
        # self.tableWidget.doubleClicked.connect(self.on_click)

        self.setGeometry(0, 0, 800, 600)
        self.setWindowTitle("NPYViewer v."+version)

        self.widget = QWidget(self)
        layout = QGridLayout()
        layout.addWidget(self.infoLb)
        layout.addWidget(self.tableWidget)
        self.widget.setLayout(layout)
        self.setCentralWidget(self.widget)
        # self.tableWidget.setItesetTextAlignmentmDelegate(AlignDelegate())

        self.layout = QVBoxLayout()

        self.setLayout(self.layout)
        self.setWindowIcon(QIcon('npyviewer_128x128.png'))

        self.show()


def main():
    if "-noGUI" not in sys.argv:
        app = QApplication(sys.argv)
        ex = MainApp()
        if len(sys.argv) ==2:
            ex.openNPY_CLI(sys.argv[1])
        sys.exit(app.exec_())
    if len(sys.argv) ==3:
        if sys.argv[2]=="-noGUI":
            print("GUI not displayed")
            openNPY_CLI_noGUI(sys.argv[1])



if __name__ == '__main__':
    main()
