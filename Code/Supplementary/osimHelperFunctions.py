# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 13:20:45 2019

@author: aafox

Set of functions that can be used to assist in processing data within OpenSim

"""

#Import necessary modules
import btk
import opensim as osim
import os
import numpy as np
from scipy import signal

# %% Function to convert c3d to trc file using btk module

def btk_c3d2trc(fullFile, markerList, filtFreq = None):
    
    #Get starting directory
    home_dir = os.getcwd()
    
    #Initialise a file reader
    reader = btk.btkAcquisitionFileReader()
    
    #Set the filename
    fileName,fileExt = os.path.splitext(os.path.basename(fullFile))
    filePath,fileName_ext = os.path.split(fullFile)
    
    #Navigate and load in file
    os.chdir(filePath)
    reader.SetFilename(fileName_ext)
    
    #Update reader
    reader.Update()
    
    #Get the btk acquisition object
    acq = reader.GetOutput()
    
    #Get the marker data frequency
    fs = acq.GetPointFrequency()
    nFrames = acq.GetPointFrameNumber()
    
    #Get marker units, start frame and end frame from metadata
    metadata = acq.GetMetaData()
    markerUnits = metadata.FindChild('POINT').value().FindChild('UNITS').value().GetInfo().ToString()
    markerUnits = str(markerUnits[0])
    startFrame = metadata.FindChild('TRIAL').value().FindChild('ACTUAL_START_FIELD').value().GetInfo().ToDouble()
    endFrame = metadata.FindChild('TRIAL').value().FindChild('ACTUAL_END_FIELD').value().GetInfo().ToDouble()
    startFrame = str(int(startFrame[0])); endFrame = str(int(endFrame[0]))
    
    #Create a dictionary to store marker data
    markers = {}
    
    #Get marker data
    for currMarker in range(0,len(markerList)):
        #Read marker point
        currPointObj = acq.GetPoint(markerList[currMarker])
        #Get marker data
        markers[markerList[currMarker]] = np.empty([nFrames,3])
        markers[markerList[currMarker]][:,0] = currPointObj.GetValues()[:,0]   #extract the first column (i.e. X)
        markers[markerList[currMarker]][:,1] = currPointObj.GetValues()[:,1]   #extract the second column (i.e. Y)
        markers[markerList[currMarker]][:,2] = currPointObj.GetValues()[:,2]   #extract the third column (i.e. Z)
    
    #Create a dictionary to store rotated marker data
    rotatedMarkers = {}
    
    #Rotate marker data from Vicon lab system to OpenSim coordinate system
    #OsimX = LabX ; OsimY = LabZ ; OsimZ = LabY * -1
    for currMarker in range(0,len(markerList)):
        #Rotate data
        rotatedMarkers[markerList[currMarker]] = np.empty([nFrames,3])
        rotatedMarkers[markerList[currMarker]][:,0] = markers[markerList[currMarker]][:,0]
        rotatedMarkers[markerList[currMarker]][:,1] = markers[markerList[currMarker]][:,2]
        rotatedMarkers[markerList[currMarker]][:,2] = markers[markerList[currMarker]][:,1] * -1
    
    #Filter marker data (if required)
    if filtFreq != None:        
        #Create low pass digital filter
        w = filtFreq / (fs / 2) #normalise filter frequency
        b,a = signal.butter(filtFreq, w, 'low')
        #Filter rotated marker data
        for currMarker in range(0,len(markerList)):
            rotatedMarkers[markerList[currMarker]][:,0] = signal.filtfilt(b, a, rotatedMarkers[markerList[currMarker]][:,0])
            rotatedMarkers[markerList[currMarker]][:,1] = signal.filtfilt(b, a, rotatedMarkers[markerList[currMarker]][:,1])
            rotatedMarkers[markerList[currMarker]][:,2] = signal.filtfilt(b, a, rotatedMarkers[markerList[currMarker]][:,2])
    
    #Output marker data
    
    #Create headers for marker names and X Y Z labels
    dataheader1 = 'Frame#\tTime\t'
    dataheader2 = '\t\t'
    for currMarker in range(0,len(markerList)):
        dataheader1 = dataheader1 + markerList[currMarker] + '\t\t\t'
        dataheader2 = dataheader2 + 'X' + str(currMarker+1) + '\t' + 'Y' + str(currMarker+1) + '\t' + 'Z' + str(currMarker+1) + '\t'
    dataheader1 = dataheader1 + '\n'
    dataheader2 = dataheader2 + '\n'
    
    #Create a new filename for data
    newFilename = (fileName + '.trc')
    
    #Initialise file
    trc = open(newFilename,'w+')
    
    #Write headers
    trc.write('PathFileType\t4\t(X/Y/Z)\t' + fullFile[0:-3] + 'trc\n')
    trc.write('DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n')
    trc.write(str(int(fs)) + '\t' + str(int(fs)) + '\t' + str(nFrames) + '\t' + str(int(len(markerList))) + '\t' + markerUnits + '\t' + str(int(fs)) + '\t' + startFrame + '\t' + endFrame + '\n')
    trc.write(dataheader1)
    trc.write(dataheader2)
    
    #Write data by row
    for nRow in range(0,nFrames-1):
        #Get current frame
        currFrame = int(startFrame) + nRow
        #Write frame and time value
        rowData = str(currFrame) + '\t' + str(int(currFrame) * (1/fs)) + '\t'
        #Get out the XYZ data for each marker for the current row of data
        for currMarker in range(0,len(markerList)):
            rowData = rowData + str(rotatedMarkers[markerList[currMarker]][nRow,0]) + '\t' + str(rotatedMarkers[markerList[currMarker]][nRow,1]) + '\t' + str(rotatedMarkers[markerList[currMarker]][nRow,2]) + '\t'
        #Append new line to current row data
        rowData = rowData + '\n'
        #Write current row
        trc.write(rowData)    
        #Cleanup
        del(currFrame,rowData)
    
    #Close file
    trc.close()
    
    #Return to starting directory
    os.chdir(home_dir)
    
# %% Function to set-up and run a scale tool
    
def setup_ScaleTool(genericModelFile = None, markerFile = None, outputDir = None,
                    modelName = 'ScaledModel', participantMass = -1, preserveMass = True,
                    measurementSetFile = None, markerPlacerFile = None,
                    printToFile = False):
    
    #Initialise scale tool
    scaler = osim.ScaleTool()
    
    #Set parameters of scale tool
    
    #Set name
    scaler.setName(modelName)
    
    #Model file
    scaler.getGenericModelMaker().setModelFileName(genericModelFile)
    
    #Output model file
    scaler.getModelScaler().setOutputModelFileName(outputDir + '\\' + modelName + '_scaledOnly.osim')
    
    #Set mass 
    ##### NEED TO DETERMINE HOW TO APPROPRIATELY SET MASS FOR THE MODEL
    ##### USE A GENERIC VALUE FOR NOW
    scaler.setSubjectMass(participantMass)
    
    #Preserve mass distribution
    scaler.getModelScaler().setPreserveMassDist(preserveMass)
    
    #Set marker file
    scaler.getModelScaler().setMarkerFileName(markerFile)
    
    #Set scaling order to use measurements
    scalingArray = osim.ArrayStr(); scalingArray.set(0,'measurements')
    scaler.getModelScaler().setScalingOrder(scalingArray)
    
    #Get the time range from the trc file
    
    ###### fix to not use TRC adapter
    
        
    trcAdapter = osim.TRCFileAdapter()
    trcData = trcAdapter.read('D:\OrthopaedicsResearch\UpperLimbPiloting\Pilot_12092019\MR\Session1\Cal_Horizontal01.trc')
    time = trcData.getIndependentColumn()
    timeArray = osim.ArrayDouble()
    timeArray.set(0,time[0]); timeArray.set(1,time[-1])
    scaler.getModelScaler().setTimeRange(timeArray)
    
    #Get measurement set from file and set
    measureObj = scaler.getModelScaler().getMeasurementSet().makeObjectFromFile(measurementSetFile)
    measureSet = osim.MeasurementSet().safeDownCast(measureObj)
    scaler.getModelScaler().setMeasurementSet(measureSet)
    
    #Get the marker placer from file
    placerObj = scaler.getMarkerPlacer().makeObjectFromFile(markerPlacerFile)
    markerPlacer = osim.MarkerPlacer().safeDownCast(placerObj)
    
    #Edit parameters of marker placer
    scaler.getMarkerPlacer().setTimeRange(timeArray)
    scaler.getMarkerPlacer().setMarkerFileName(markerFile)
    scaler.getMarkerPlacer().setOutputMotionFileName(outputDir + '\\' + modelName + '_static_output.mot')
    scaler.getMarkerPlacer().setOutputModelFileName(outputDir + '\\' + modelName + '_scaledAdjusted.osim')
    
    #Get and add the IK tasks from the loaded in marker placer
    #This now also includes IK coordinate tasks with high weights that ask the scale
    #tool to use the default values for the coordinates. These match up to the pose
    #that we get the participant to do in a horizontal arm calibration trial.
    #Only uses the default value for some coordinates, as some can't be exactly
    #confident in and this messes with model marker placement (e.g. elevation plane)
    taskSet = markerPlacer.getIKTaskSet()
    for nTask in range(0,taskSet.getSize()-1):
        #Add task
        scaler.getMarkerPlacer().getIKTaskSet().adoptAndAppend(taskSet.get(nTask))
    
    #Print scale tool to file
    if printToFile:
        scaler.printToXML(outputDir + '\\' + modelName + '_SetupScale.xml')
    
    #Run scale tool
    scaler.run()
    
# %%    



