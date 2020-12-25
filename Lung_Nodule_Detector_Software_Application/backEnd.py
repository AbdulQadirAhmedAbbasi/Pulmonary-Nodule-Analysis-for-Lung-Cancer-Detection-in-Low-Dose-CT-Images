#BackEnd

#Importing Libraries
import sys
import csv
import os

import pyqtgraph as pg
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt


from PyQt4 import QtGui, QtCore
from frontEnd import Ui_MainWindow
from PIL import Image


class Main(QtGui.QMainWindow):
    
    def __init__(self):
        QtGui.QMainWindow.__init__(self)        
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        #Setting icons to menu bar & main window
        self.setWindowIcon(QtGui.QIcon("resources/lungs.png"))
        self.ui.actionOpen.setIcon(QtGui.QIcon("resources/open.png"))
        self.ui.actionEqualize_Histogram.setIcon(QtGui.QIcon("resources/EqualizeHistogram.png"))
        self.ui.actionCancel_Equalize_Histogram.setIcon(QtGui.QIcon("resources/DeEqualizeHistogram.png"))
        self.ui.actionPlay.setIcon(QtGui.QIcon("resources/play.png"))
        self.ui.actionStop.setIcon(QtGui.QIcon("resources/stop.png"))
        self.ui.actionReplay.setIcon(QtGui.QIcon("resources/replay.png"))
        self.ui.actionClose.setIcon(QtGui.QIcon("resources/close.png"))
        self.ui.actionQuit.setIcon(QtGui.QIcon("resources/quit.png"))
        self.ui.actionExtract_Nodules.setIcon(QtGui.QIcon("resources/extract.png"))
        
        #Adding Actions to File Menu in Menu Bar
        self.ui.actionOpen.triggered.connect(self.open_file)
        self.ui.actionEqualize_Histogram.triggered.connect(self.equalizeHistogram_file)
        self.ui.actionCancel_Equalize_Histogram.triggered.connect(self.de_equalizeHistogram_file)
        self.ui.actionPlay.triggered.connect(self.play_file)
        self.ui.actionStop.triggered.connect(self.stop_file)
        self.ui.actionReplay.triggered.connect(self.replay_file)
        self.ui.actionClose.triggered.connect(self.close_file)
        self.ui.actionQuit.triggered.connect(self.quit_application)
        self.ui.actionExtract_Nodules.triggered.connect(self.extract_nodules)
        
        
        #Tool Bar

        #Opening MHD file action        
        self.ToolbarActionOpen = QtGui.QAction(QtGui.QIcon("resources/open.png"), "Open", self)
        self.ToolbarActionOpen.setStatusTip("Open MHD File")
        self.ToolbarActionOpen.triggered.connect(self.open_file)

        #Closing MHD file action        
        self.ToolbarActionClose = QtGui.QAction(QtGui.QIcon("resources/close.png"), "Close", self)
        self.ToolbarActionClose.setStatusTip("Close MHD File")
        self.ToolbarActionClose.triggered.connect(self.close_file)
        self.ToolbarActionClose.setEnabled(False)
        
        #Equalizing histogram for MHD file action
        self.ToolbarActionEqualizeHistogram = QtGui.QAction(QtGui.QIcon("resources/equalizeHistogram.png"), "Equalize Histogram", self)
        self.ToolbarActionEqualizeHistogram.setStatusTip("Histogram Equaliztion for MHD File")
        self.ToolbarActionEqualizeHistogram.triggered.connect(self.equalizeHistogram_file)
        self.ToolbarActionEqualizeHistogram.setEnabled(False)
 
        #De Equalizing histogram for MHD file action
        self.ToolbarActionDeEqualizeHistogram = QtGui.QAction(QtGui.QIcon("resources/DeEqualizeHistogram.png"), "Cancel Equalize Histogram", self)
        self.ToolbarActionDeEqualizeHistogram.setStatusTip("Cancel Histogram Equalization Effect on MHD File")
        self.ToolbarActionDeEqualizeHistogram.triggered.connect(self.de_equalizeHistogram_file)
        self.ToolbarActionDeEqualizeHistogram.setEnabled(False)

       
        #Auto playing MHD file action
        self.ToolbarActionAutoPlay = QtGui.QAction(QtGui.QIcon("resources/play.png"), "Play", self)
        self.ToolbarActionAutoPlay.setStatusTip("Start Playing MHD File")
        self.ToolbarActionAutoPlay.triggered.connect(self.play_file)
        self.ToolbarActionAutoPlay.setEnabled(False)
 

       #Stoping MHD file action
        self.ToolbarActionStopPlay = QtGui.QAction(QtGui.QIcon("resources/stop.png"), "Stop", self)
        self.ToolbarActionStopPlay.setStatusTip("Stop Playing MHD File")
        self.ToolbarActionStopPlay.triggered.connect(self.stop_file)
        self.ToolbarActionStopPlay.setEnabled(False)
        
       #Replaying MHD file action
        self.ToolbarActionRePlay = QtGui.QAction(QtGui.QIcon("resources/replay.png"), "Replay", self)
        self.ToolbarActionRePlay.setStatusTip("Replay MHD File")
        self.ToolbarActionRePlay.triggered.connect(self.replay_file)
        self.ToolbarActionRePlay.setEnabled(False)


         #Extracting Nodules from MHD file action
        self.ToolbarActionExtractNodules = QtGui.QAction(QtGui.QIcon("resources/extract.png"), "Extract Nodules", self)
        self.ToolbarActionExtractNodules.setStatusTip("Extract Nodules from MHD File")
        self.ToolbarActionExtractNodules.triggered.connect(self.extract_nodules)
        self.ToolbarActionExtractNodules.setEnabled(False)

        #Adding Actions to Toolbar
        self.ui.toolBar.addAction(self.ToolbarActionOpen)
        self.ui.toolBar.addAction(self.ToolbarActionClose)
        self.ui.toolBar.addSeparator()
        
        self.ui.toolBar.addAction(self.ToolbarActionEqualizeHistogram)
        self.ui.toolBar.addAction(self.ToolbarActionDeEqualizeHistogram)
        self.ui.toolBar.addAction(self.ToolbarActionExtractNodules)
        self.ui.toolBar.addSeparator()
        
        self.ui.toolBar.addAction(self.ToolbarActionAutoPlay)
        self.ui.toolBar.addAction(self.ToolbarActionStopPlay)
        self.ui.toolBar.addAction(self.ToolbarActionRePlay)
        self.ui.toolBar.addSeparator()
        
        
        #GlobalVariables
        self.HistogramFlag = 0;
        self.patchplot = 0;
        
        
    #Function for loading MHD image & converting it into numpy Array
    def load_itk_image(self,filename):
        itkimage = sitk.ReadImage(filename)
        numpyImage = sitk.GetArrayFromImage(itkimage)
        numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
        numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))
        return numpyImage, numpyOrigin, numpySpacing
    
    #Function for reading CSV file    
    def readCSV(self, filename):
        lines = []
        with open(filename, "rb") as f:
          csvreader = csv.reader(f)
          for line in csvreader:
            lines.append(line)
          return lines
    #Function for converting World to Voxel coordinates
    def worldToVoxelCoord(self, worldCoord, origin, spacing):
        stretchedVoxelCoord = np.absolute(worldCoord - origin)
        voxelCoord = stretchedVoxelCoord / spacing
        return voxelCoord
    
    #Function for normalizing planes 
    def normalizePlanes(self, npzarray):
        maxHU = 400.
        minHU = -1000.
        npzarray = (npzarray - minHU) / (maxHU - minHU)
        npzarray[npzarray>1] = 1.
        npzarray[npzarray<0] = 0.
        return npzarray


    #Function for applying Histogram Equilization to MHD image
    def image_histogram_equalization(self ,image, number_bins=256):
    
        # get image histogram
        image_histogram, bins = np.histogram(image.flatten(), number_bins, normed=True)
        cdf = image_histogram.cumsum() # cumulative distribution function
        cdf = 255 * cdf / cdf[-1] # normalize

        # use linear interpolation of cdf to find new pixel values
        image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

        return image_equalized.reshape(image.shape), cdf

    #Function for playing file for "ToolbarActionAutoPlay            
    def play_file(self):
        
        #Enabling Title Bar
        self.ToolbarActionAutoPlay.setEnabled(False)
        self.ToolbarActionStopPlay.setEnabled(True)

        #Enabling Menu Bar
        self.ui.actionPlay.setEnabled(False)
        self.ui.actionStop.setEnabled(True)        
        
        #Enabling Tool Bar
        self.ui.graphicsView.play(20)
        self.ToolbarActionRePlay.setEnabled(True) 
        
        #Enabling Menu Bar
        self.ui.actionReplay.setEnabled(True)
    
        self.play = 1


    #Function for playing file for "ToolbarActionAutoPlay            
    def replay_file(self):
        
        #Enabling Tool Bar
        self.ToolbarActionAutoPlay.setEnabled(False)
        self.ToolbarActionRePlay.setEnabled(True)        
        self.ToolbarActionStopPlay.setEnabled(True)
        
        #Enabling Menu Bar
        self.ui.actionPlay.setEnabled(False)
        self.ui.actionReplay.setEnabled(True)
        self.ui.actionStop.setEnabled(True)
        
        self.ui.graphicsView.setImage(np.swapaxes(self.numpyImage,1,2),autoRange=True, autoLevels=True, levels=None, axes=None, xvals=None, pos=None, scale=None, transform=None, autoHistogramRange=True)
        self.ui.graphicsView.play(20)    
    
        if self.HistogramFlag == 1:
            self.ui.graphicsView.setImage(np.swapaxes(self.data_equalized,1,2),autoRange=True, autoLevels=True, levels=None, axes=None, xvals=None, pos=None, scale=None, transform=None, autoHistogramRange=True)
            self.ui.graphicsView.play(20)
        else:    
            self.ui.graphicsView.setImage(np.swapaxes(self.numpyImage,1,2),autoRange=True, autoLevels=True, levels=None, axes=None, xvals=None, pos=None, scale=None, transform=None, autoHistogramRange=True)
            self.ui.graphicsView.play(20)
        
    #Function for stoping file for "ToolbarActionStopPlay            
    def stop_file(self):
        
        #Enabling Tool Bar
        self.ToolbarActionRePlay.setEnabled(False)

        #Enabling Menu Bar
        self.ui.actionReplay.setEnabled(False)

        if self.play == 1:
            self.ui.graphicsView.setImage(np.swapaxes(self.numpyImage,1,2),autoRange=True, autoLevels=True, levels=None, axes=None, xvals=None, pos=None, scale=None, transform=None, autoHistogramRange=True)
            
            #Enabling Tool Bar
            self.ToolbarActionAutoPlay.setEnabled(True)
            self.ToolbarActionStopPlay.setEnabled(False)
            
            #Enabling Menu Bar
            self.ui.actionPlay.setEnabled(True)
            self.ui.actionStop.setEnabled(False)
            
            if self.HistogramFlag == 1:
                self.ui.graphicsView.setImage(np.swapaxes(self.data_equalized,1,2),autoRange=True, autoLevels=True, levels=None, axes=None, xvals=None, pos=None, scale=None, transform=None, autoHistogramRange=True)
            else:    
                self.ui.graphicsView.setImage(np.swapaxes(self.numpyImage,1,2),autoRange=True, autoLevels=True, levels=None, axes=None, xvals=None, pos=None, scale=None, transform=None, autoHistogramRange=True)
        
    #Function for closing MHD file 
    def close_file(self):     
        
        self.HistogramFlag = 0
        self.play = 0
        #Enabling Tool Bar Actions        
        self.ToolbarActionEqualizeHistogram.setEnabled(False) 
        self.ToolbarActionDeEqualizeHistogram.setEnabled(False) 
        self.ToolbarActionAutoPlay.setEnabled(False)
        self.ToolbarActionStopPlay.setEnabled(False)
        self.ToolbarActionRePlay.setEnabled(False)
        self.ToolbarActionClose.setEnabled(False)
        self.ToolbarActionExtractNodules.setEnabled(False)
        
        #Enabling Menu Bar Actions        
        self.ui.actionEqualize_Histogram.setEnabled(False)
        self.ui.actionCancel_Equalize_Histogram.setEnabled(False)
        self.ui.actionPlay.setEnabled(False)
        self.ui.actionStop.setEnabled(False)
        self.ui.actionReplay.setEnabled(False)
        self.ui.actionClose.setEnabled(False)        
        self.ui.actionExtract_Nodules.setEnabled(False)

        #Clearing3D plotting of CT scan            
        self.ui.graphicsView.clear()
        
        
    #Function for quiting Application
    def quit_application(self):
        #message box
        choice = QtGui.QMessageBox.question(self,"Quit Application",
                                            "Are you sure you want to quit application?",
                                            QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)
        if choice == QtGui.QMessageBox.Yes:
            QtCore.QCoreApplication.instance().quit()
        else:
            pass
     
    #Function for equalizing file for "ToolbarActionEqualizeHistogram"    
    def equalizeHistogram_file(self):
        
        self.HistogramFlag = 1
        #Enabling Tool Bar Actions        
        self.ToolbarActionEqualizeHistogram.setEnabled(False) 
        self.ToolbarActionDeEqualizeHistogram.setEnabled(True) 
        self.ToolbarActionAutoPlay.setEnabled(True)
        self.ToolbarActionStopPlay.setEnabled(False)
        self.ToolbarActionExtractNodules.setEnabled(False)
    
        #Enabling Menu Bar Actions        
        self.ui.actionEqualize_Histogram.setEnabled(False)
        self.ui.actionCancel_Equalize_Histogram.setEnabled(True)
        self.ui.actionPlay.setEnabled(True)
        self.ui.actionStop.setEnabled(False)
        self.ui.actionExtract_Nodules.setEnabled(False)
        
        self.ui.graphicsView.setImage(np.swapaxes(self.data_equalized,1,2),autoRange=True, autoLevels=True, levels=None, axes=None, xvals=None, pos=None, scale=None, transform=None, autoHistogramRange=True)
        #self.ToolbarActionEqualizeHistogram.triggered.connect(self.de_equalizeHistogram_file)
        
    def de_equalizeHistogram_file(self):
        
        self.HistogramFlag = 0
        #Enabling Tool Bar Actions
        self.ToolbarActionEqualizeHistogram.setEnabled(True)
        self.ToolbarActionDeEqualizeHistogram.setEnabled(False) 
        self.ToolbarActionAutoPlay.setEnabled(True)
        self.ToolbarActionStopPlay.setEnabled(False)
        self.ToolbarActionExtractNodules.setEnabled(True)
        
        #Enabling Menu Bar Actions
        self.ui.actionEqualize_Histogram.setEnabled(True)
        self.ui.actionCancel_Equalize_Histogram.setEnabled(False)
        self.ui.actionPlay.setEnabled(True)
        self.ui.actionStop.setEnabled(False)
        self.ui.actionExtract_Nodules.setEnabled(True)
        
        self.ui.graphicsView.setImage(np.swapaxes(self.numpyImage,1,2),autoRange=True, autoLevels=True, levels=None, axes=None, xvals=None, pos=None, scale=None, transform=None, autoHistogramRange=True)


    #Function for extracting Nodules for "actionExtract_Nodules"    
    def extract_nodules(self):
        #Showing Busy Status While Opening        
        self.ui.statusbar.showMessage("Status | Busy",8000)
        self.ui.tabResults.setEnabled(True)
        
        #index = self.ui.tabWidget.currentIndex()
        #print(index)

        #Changing current index to move to next widget        
        self.ui.tabWidget.setCurrentIndex(1)
        
        self.cand_path = 'data/candidates.csv'
        
        
        # load candidates
        cands = self.readCSV(self.cand_path)

        
        # get candidates
        for cand in cands[1:]:
            worldCoord = np.asarray([float(cand[3]),float(cand[2]),float(cand[1])])
            voxelCoord = self.worldToVoxelCoord(worldCoord, self.numpyOrigin, self.numpySpacing)
            voxelWidth = 65
            
        for cand in cands[1:]:
            worldCoord = np.asarray([float(cand[3]),float(cand[2]),float(cand[1])])
            voxelCoord = self.worldToVoxelCoord(worldCoord, self.numpyOrigin, self.numpySpacing)
            voxelWidth = 65

            patch = self.numpyImage[int(voxelCoord[0]),int(voxelCoord[1]-voxelWidth/2):int(voxelCoord[1]+voxelWidth/2),int(voxelCoord[2]-voxelWidth/2):int(voxelCoord[2]+voxelWidth/2)]            
            
            #patch = self.numpyImage[voxelCoord[0],voxelCoord[1]-voxelWidth/2:voxelCoord[1]+voxelWidth/2,voxelCoord[2]-voxelWidth/2:voxelCoord[2]+voxelWidth/2]
            patch = self.normalizePlanes(patch)
            
            #Directory for Storing Results
            outputDir = 'patches/'
            
            plt.imshow(patch, cmap='gray')
            plt.show()
            
            Image.fromarray(patch*255).convert('L').save(os.path.join(outputDir, 'patch_' + str(worldCoord[0]) + '_' + str(worldCoord[1]) + '_' + str(worldCoord[2]) + '.tiff'))

        self.ui.statusbar.showMessage("Status | Complete",8000)
     
    #Function for opening file for "actionOpen"    
    def open_file(self):
        
        
        
        name = QtGui.QFileDialog.getOpenFileName(self,"Open File Dialouge","","Image files (*.mhd)")
        img_path = str(name) #must convert path name into string other wise simple ITK will give error       
        
        if img_path:
            
            #Showing Busy Status While Opening        
            self.ui.statusbar.showMessage("Status | Busy",8000)            
            
            #Showing tabWidget
            self.ui.tabWidget.setEnabled(True)
            self.ui.tabViewer.setEnabled(True)
                        
            self.numpyImage, self.numpyOrigin, self.numpySpacing = self.load_itk_image(img_path)
            
            #Normalizing MHD Image 
            self.normalizedNumpyImage = self.normalizePlanes(self.numpyImage)            
            
            #Enabling Actions for opened MHD File

            #Tool Bar
            self.ToolbarActionEqualizeHistogram.setEnabled(True)            
            self.ToolbarActionAutoPlay.setEnabled(True)
            self.ToolbarActionClose.setEnabled(True)
            self.ToolbarActionExtractNodules.setEnabled(True)
            
            #Menu Bar
            self.ui.actionEqualize_Histogram.setEnabled(True)
            self.ui.actionPlay.setEnabled(True)
            self.ui.actionClose.setEnabled(True)
            self.ui.actionExtract_Nodules.setEnabled(True)
            
            # loop for histogram equilization
            self.data_equalized = np.zeros(self.numpyImage.shape)
            for i in range(self.numpyImage.shape[0]):
                image = self.numpyImage[i, :, :]
                self.data_equalized[i, :, :] = self.image_histogram_equalization(image)[0]
            
                            
            #3D plotting of CT scan
            self.ui.graphicsView.show()
            
            
            #numpy arrays swapped for horizontal visualization
            self.ui.graphicsView.setImage(np.swapaxes(self.numpyImage,1,2),autoRange=True, autoLevels=True, levels=None, axes=None, xvals=None, pos=None, scale=None, transform=None, autoHistogramRange=True)
            self.ui.graphicsView.setCursor(QtCore.Qt.CrossCursor)            
            self.ui.statusbar.showMessage("Status | Ready",8000)

        
if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    window = Main()
    window.show()
    sys.exit(app.exec_())
       