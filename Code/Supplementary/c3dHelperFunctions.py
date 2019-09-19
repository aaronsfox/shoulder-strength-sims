# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 11:17:55 2019

@author: aafox

Set of functions that can be used to assist in processing C3D data

"""

#Import necessary modules
import btk
import opensim as osim
import os
import math
import numpy as np
from scipy import signal
from scipy.optimize import fsolve
import lxml.etree as ET
import subprocess

# %% Function to add shoulder and elbow joint centres

def addFunctionalJointCentres(trialC3D, shoulderC3D = None, elbowC3D = None, filtFreq = None):
    
    if shoulderC3D != None:
        
        #Add shoulder joint centre marker
    
        #Initialise a file reader
        reader = btk.btkAcquisitionFileReader()
        
        #Start with the shoulder joint centre
        
        #Load in SCoRE c3d file
        reader.SetFilename(shoulderC3D)
        
        #Update reader
        reader.Update()
        
        #Get the btk acquisition object
        acq = reader.GetOutput()
        
        #Get number of frames
        nFrames = acq.GetPointFrameNumber()
        
        #Get the shoulder joint centre marker
        SJCmrkr = np.empty([nFrames,3])
        SJCmrkr[:,0] = acq.GetPoint('Thorax_UpperArm.R_score').GetValues()[:,0]
        SJCmrkr[:,1] = acq.GetPoint('Thorax_UpperArm.R_score').GetValues()[:,1]
        SJCmrkr[:,2] = acq.GetPoint('Thorax_UpperArm.R_score').GetValues()[:,2]
        
        #Get the three markers to use to calculate joint centre position
        #These will be: R.PUA.Top, R.LUA.Top, R.LUA.Right
        Amrkr = np.empty([nFrames,3])
        Amrkr[:,0] = acq.GetPoint('R.PUA.Top').GetValues()[:,0]
        Amrkr[:,1] = acq.GetPoint('R.PUA.Top').GetValues()[:,1]
        Amrkr[:,2] = acq.GetPoint('R.PUA.Top').GetValues()[:,2]
        Bmrkr = np.empty([nFrames,3])
        Bmrkr[:,0] = acq.GetPoint('R.LUA.Top').GetValues()[:,0]
        Bmrkr[:,1] = acq.GetPoint('R.LUA.Top').GetValues()[:,1]
        Bmrkr[:,2] = acq.GetPoint('R.LUA.Top').GetValues()[:,2]
        Cmrkr = np.empty([nFrames,3])
        Cmrkr[:,0] = acq.GetPoint('R.LUA.Right').GetValues()[:,0]
        Cmrkr[:,1] = acq.GetPoint('R.LUA.Right').GetValues()[:,1]
        Cmrkr[:,2] = acq.GetPoint('R.LUA.Right').GetValues()[:,2]
        
        #Filter marker data (if required)
        if filtFreq != None:
            #Get sampling frequency
            fs = acq.GetPointFrequency()            
            #Create low pass digital filter
            w = filtFreq / (fs / 2) #normalise filter frequency
            b,a = signal.butter(filtFreq, w, 'low')
            #Filter rotated marker data
            SJCmrkr[:,0] = signal.filtfilt(b, a, SJCmrkr[:,0])
            SJCmrkr[:,1] = signal.filtfilt(b, a, SJCmrkr[:,1])
            SJCmrkr[:,2] = signal.filtfilt(b, a, SJCmrkr[:,2])
            Amrkr[:,0] = signal.filtfilt(b, a, Amrkr[:,0])
            Amrkr[:,1] = signal.filtfilt(b, a, Amrkr[:,1])
            Amrkr[:,2] = signal.filtfilt(b, a, Amrkr[:,2])
            Bmrkr[:,0] = signal.filtfilt(b, a, Bmrkr[:,0])
            Bmrkr[:,1] = signal.filtfilt(b, a, Bmrkr[:,1])
            Bmrkr[:,2] = signal.filtfilt(b, a, Bmrkr[:,2])
            Cmrkr[:,0] = signal.filtfilt(b, a, Cmrkr[:,0])
            Cmrkr[:,1] = signal.filtfilt(b, a, Cmrkr[:,1])
            Cmrkr[:,2] = signal.filtfilt(b, a, Cmrkr[:,2])
        
        #Calculate the distance between the markers at each frame
        distA = np.empty([nFrames,1]); distB = np.empty([nFrames,1]); distC = np.empty([nFrames,1])
        for p in range(0,nFrames-1):
            distA[p,0] = math.sqrt((SJCmrkr[p,0] - Amrkr[p,0])**2 + (SJCmrkr[p,1] - Amrkr[p,1])**2 + (SJCmrkr[p,2] - Amrkr[p,2])**2)
            distB[p,0] = math.sqrt((SJCmrkr[p,0] - Bmrkr[p,0])**2 + (SJCmrkr[p,1] - Bmrkr[p,1])**2 + (SJCmrkr[p,2] - Bmrkr[p,2])**2)
            distC[p,0] = math.sqrt((SJCmrkr[p,0] - Cmrkr[p,0])**2 + (SJCmrkr[p,1] - Cmrkr[p,1])**2 + (SJCmrkr[p,2] - Cmrkr[p,2])**2)
        
        #Calculate average distances
        distA_avg = np.mean(distA); distB_avg = np.mean(distB); distC_avg = np.mean(distC)
        
        #Read in the experimental data c3d file
        readerExp = btk.btkAcquisitionFileReader()
        
        #Load in experimental c3d trial
        readerExp.SetFilename(trialC3D)
        
        #Update reader
        readerExp.Update()
        
        #Get the btk acquisition object
        acqExp = readerExp.GetOutput()
        
        #Get number of frames
        nFramesExp = acqExp.GetPointFrameNumber()
        
        #Extract the same markers used for calculating distance from the experimental data
        AmrkrExp = np.empty([nFramesExp,3])
        AmrkrExp[:,0] = acqExp.GetPoint('R.PUA.Top').GetValues()[:,0]
        AmrkrExp[:,1] = acqExp.GetPoint('R.PUA.Top').GetValues()[:,1]
        AmrkrExp[:,2] = acqExp.GetPoint('R.PUA.Top').GetValues()[:,2]
        BmrkrExp = np.empty([nFramesExp,3])
        BmrkrExp[:,0] = acqExp.GetPoint('R.LUA.Top').GetValues()[:,0]
        BmrkrExp[:,1] = acqExp.GetPoint('R.LUA.Top').GetValues()[:,1]
        BmrkrExp[:,2] = acqExp.GetPoint('R.LUA.Top').GetValues()[:,2]
        CmrkrExp = np.empty([nFramesExp,3])
        CmrkrExp[:,0] = acqExp.GetPoint('R.LUA.Right').GetValues()[:,0]
        CmrkrExp[:,1] = acqExp.GetPoint('R.LUA.Right').GetValues()[:,1]
        CmrkrExp[:,2] = acqExp.GetPoint('R.LUA.Right').GetValues()[:,2]
        
        #Filter marker data (if required)
        if filtFreq != None:
            #Get sampling frequency
            fs = acq.GetPointFrequency()            
            #Create low pass digital filter
            w = filtFreq / (fs / 2) #normalise filter frequency
            b,a = signal.butter(filtFreq, w, 'low')
            #Filter rotated marker data
            AmrkrExp[:,0] = signal.filtfilt(b, a, AmrkrExp[:,0])
            AmrkrExp[:,1] = signal.filtfilt(b, a, AmrkrExp[:,1])
            AmrkrExp[:,2] = signal.filtfilt(b, a, AmrkrExp[:,2])
            BmrkrExp[:,0] = signal.filtfilt(b, a, BmrkrExp[:,0])
            BmrkrExp[:,1] = signal.filtfilt(b, a, BmrkrExp[:,1])
            BmrkrExp[:,2] = signal.filtfilt(b, a, BmrkrExp[:,2])
            CmrkrExp[:,0] = signal.filtfilt(b, a, CmrkrExp[:,0])
            CmrkrExp[:,1] = signal.filtfilt(b, a, CmrkrExp[:,1])
            CmrkrExp[:,2] = signal.filtfilt(b, a, CmrkrExp[:,2])
        
        #Use the experimental marker data and calculated distances to solve the distance
        #equations for the XYZ positions of the shoulder joint centre in the experimental
        #trial. For the first frame, we'll use a marker that is close to the propose
        #joint centre (i.e. R.ACR)
        
        #Define the initial guess for shoulder joint centre on the first frame
        initialGuess = [acqExp.GetPoint('R.ACR').GetValues()[0,0],
                        acqExp.GetPoint('R.ACR').GetValues()[0,1],
                        acqExp.GetPoint('R.ACR').GetValues()[0,2]]
        
        #Initialise empty array for SJC experimental data
        SJCmrkrExp = np.empty([nFramesExp,3])
        
        #Loop through experimental trial frames and calculate SJC position
        for frameNo in range(0,nFramesExp):
            
            #Define the marker positions for below solver equations
            xA = AmrkrExp[frameNo,0]; yA = AmrkrExp[frameNo,1]; zA = AmrkrExp[frameNo,2];
            xB = BmrkrExp[frameNo,0]; yB = BmrkrExp[frameNo,1]; zB = BmrkrExp[frameNo,2];
            xC = CmrkrExp[frameNo,0]; yC = CmrkrExp[frameNo,1]; zC = CmrkrExp[frameNo,2];
            
            #Define a function that expresses the equations
            def f(p):
                x,y,z = p
                #Define functions
                fA = ((x - xA)**2 + (y - yA)**2 + (z - zA)**2) - distA_avg**2
                fB = ((x - xB)**2 + (y - yB)**2 + (z - zB)**2) - distB_avg**2
                fC = ((x - xC)**2 + (y - yC)**2 + (z - zC)**2) - distC_avg**2
                return [fA,fB,fC]
            
            #Test function with values
            #print(f([30,40,50]))
            
            #Check whether to update initial guess
            if frameNo != 0:
                #Update initial guess
                initialGuess = [SJCmrkrExp[frameNo-1,0],SJCmrkrExp[frameNo-1,1],SJCmrkrExp[frameNo-1,2]]
                
            #Solve equation for current marker position
            SJCmrkrExp[frameNo,0],SJCmrkrExp[frameNo,1],SJCmrkrExp[frameNo,2] = fsolve(f,initialGuess)
            
            #Cleanup
            #del(xA,xB,xC,yA,yB,yC,zA,zB,zC)
        
        #Add new marker back into c3d data as R.SJC
            
        #Create new empty point
        newPoint = btk.btkPoint(acqExp.GetPointFrameNumber())
        #Set label on new point
        newPoint.SetLabel('R.SJC')
        #Set values for new marker from those calculated
        newPoint.SetValues(SJCmrkrExp)
        #Append new point to acquisition object
        acqExp.AppendPoint(newPoint)
    
    #Re-run the above process to do the elbow joint centre
    
    if elbowC3D != None:
        
        #Initialise a file reader
        reader = btk.btkAcquisitionFileReader()
        
        #Start with the shoulder joint centre
        
        #Load in SCoRE c3d file
        reader.SetFilename(elbowC3D)
        
        #Update reader
        reader.Update()
        
        #Get the btk acquisition object
        acq = reader.GetOutput()
        
        #Get number of frames
        nFrames = acq.GetPointFrameNumber()
        
        #Get the shoulder joint centre marker
        EJCmrkr = np.empty([nFrames,3])
        EJCmrkr[:,0] = acq.GetPoint('UpperArm.R_ForeArm.R_score').GetValues()[:,0]
        EJCmrkr[:,1] = acq.GetPoint('UpperArm.R_ForeArm.R_score').GetValues()[:,1]
        EJCmrkr[:,2] = acq.GetPoint('UpperArm.R_ForeArm.R_score').GetValues()[:,2]
        
        #Get the three markers to use to calculate joint centre position
        #These will be: R.FA.Top, R.FA.Left, R.PUA.Right
        Amrkr = np.empty([nFrames,3])
        Amrkr[:,0] = acq.GetPoint('R.FA.Top').GetValues()[:,0]
        Amrkr[:,1] = acq.GetPoint('R.FA.Top').GetValues()[:,1]
        Amrkr[:,2] = acq.GetPoint('R.FA.Top').GetValues()[:,2]
        Bmrkr = np.empty([nFrames,3])
        Bmrkr[:,0] = acq.GetPoint('R.FA.Left').GetValues()[:,0]
        Bmrkr[:,1] = acq.GetPoint('R.FA.Left').GetValues()[:,1]
        Bmrkr[:,2] = acq.GetPoint('R.FA.Left').GetValues()[:,2]
        Cmrkr = np.empty([nFrames,3])
        Cmrkr[:,0] = acq.GetPoint('R.PUA.Right').GetValues()[:,0]
        Cmrkr[:,1] = acq.GetPoint('R.PUA.Right').GetValues()[:,1]
        Cmrkr[:,2] = acq.GetPoint('R.PUA.Right').GetValues()[:,2]
        
        #Filter marker data (if required)
        if filtFreq != None:
            #Get sampling frequency
            fs = acq.GetPointFrequency()            
            #Create low pass digital filter
            w = filtFreq / (fs / 2) #normalise filter frequency
            b,a = signal.butter(filtFreq, w, 'low')
            #Filter rotated marker data
            EJCmrkr[:,0] = signal.filtfilt(b, a, EJCmrkr[:,0])
            EJCmrkr[:,1] = signal.filtfilt(b, a, EJCmrkr[:,1])
            EJCmrkr[:,2] = signal.filtfilt(b, a, EJCmrkr[:,2])
            Amrkr[:,0] = signal.filtfilt(b, a, Amrkr[:,0])
            Amrkr[:,1] = signal.filtfilt(b, a, Amrkr[:,1])
            Amrkr[:,2] = signal.filtfilt(b, a, Amrkr[:,2])
            Bmrkr[:,0] = signal.filtfilt(b, a, Bmrkr[:,0])
            Bmrkr[:,1] = signal.filtfilt(b, a, Bmrkr[:,1])
            Bmrkr[:,2] = signal.filtfilt(b, a, Bmrkr[:,2])
            Cmrkr[:,0] = signal.filtfilt(b, a, Cmrkr[:,0])
            Cmrkr[:,1] = signal.filtfilt(b, a, Cmrkr[:,1])
            Cmrkr[:,2] = signal.filtfilt(b, a, Cmrkr[:,2])
        
        #Calculate the distance between the markers at each frame
        distA = np.empty([nFrames,1]); distB = np.empty([nFrames,1]); distC = np.empty([nFrames,1])
        for p in range(0,nFrames-1):
            distA[p,0] = math.sqrt((EJCmrkr[p,0] - Amrkr[p,0])**2 + (EJCmrkr[p,1] - Amrkr[p,1])**2 + (EJCmrkr[p,2] - Amrkr[p,2])**2)
            distB[p,0] = math.sqrt((EJCmrkr[p,0] - Bmrkr[p,0])**2 + (EJCmrkr[p,1] - Bmrkr[p,1])**2 + (EJCmrkr[p,2] - Bmrkr[p,2])**2)
            distC[p,0] = math.sqrt((EJCmrkr[p,0] - Cmrkr[p,0])**2 + (EJCmrkr[p,1] - Cmrkr[p,1])**2 + (EJCmrkr[p,2] - Cmrkr[p,2])**2)
        
        #Calculate average distances
        distA_avg = np.mean(distA); distB_avg = np.mean(distB); distC_avg = np.mean(distC)
        
        #Extract the same markers used for calculating distance from the experimental data
        AmrkrExp = np.empty([nFramesExp,3])
        AmrkrExp[:,0] = acqExp.GetPoint('R.FA.Top').GetValues()[:,0]
        AmrkrExp[:,1] = acqExp.GetPoint('R.FA.Top').GetValues()[:,1]
        AmrkrExp[:,2] = acqExp.GetPoint('R.FA.Top').GetValues()[:,2]
        BmrkrExp = np.empty([nFramesExp,3])
        BmrkrExp[:,0] = acqExp.GetPoint('R.FA.Left').GetValues()[:,0]
        BmrkrExp[:,1] = acqExp.GetPoint('R.FA.Left').GetValues()[:,1]
        BmrkrExp[:,2] = acqExp.GetPoint('R.FA.Left').GetValues()[:,2]
        CmrkrExp = np.empty([nFramesExp,3])
        CmrkrExp[:,0] = acqExp.GetPoint('R.PUA.Right').GetValues()[:,0]
        CmrkrExp[:,1] = acqExp.GetPoint('R.PUA.Right').GetValues()[:,1]
        CmrkrExp[:,2] = acqExp.GetPoint('R.PUA.Right').GetValues()[:,2]
        
        #Filter marker data (if required)
        if filtFreq != None:
            #Get sampling frequency
            fs = acq.GetPointFrequency()            
            #Create low pass digital filter
            w = filtFreq / (fs / 2) #normalise filter frequency
            b,a = signal.butter(filtFreq, w, 'low')
            #Filter rotated marker data
            AmrkrExp[:,0] = signal.filtfilt(b, a, AmrkrExp[:,0])
            AmrkrExp[:,1] = signal.filtfilt(b, a, AmrkrExp[:,1])
            AmrkrExp[:,2] = signal.filtfilt(b, a, AmrkrExp[:,2])
            BmrkrExp[:,0] = signal.filtfilt(b, a, BmrkrExp[:,0])
            BmrkrExp[:,1] = signal.filtfilt(b, a, BmrkrExp[:,1])
            BmrkrExp[:,2] = signal.filtfilt(b, a, BmrkrExp[:,2])
            CmrkrExp[:,0] = signal.filtfilt(b, a, CmrkrExp[:,0])
            CmrkrExp[:,1] = signal.filtfilt(b, a, CmrkrExp[:,1])
            CmrkrExp[:,2] = signal.filtfilt(b, a, CmrkrExp[:,2])
        
        #Use the experimental marker data and calculated distances to solve the distance
        #equations for the XYZ positions of the shoulder joint centre in the experimental
        #trial. For the first frame, we'll use a marker that is close to the propose
        #joint centre (i.e. R.PUA.Left)
        
        #Define the initial guess for shoulder joint centre on the first frame
        initialGuess = [acqExp.GetPoint('R.PUA.Left').GetValues()[0,0],
                        acqExp.GetPoint('R.PUA.Left').GetValues()[0,1],
                        acqExp.GetPoint('R.PUA.Left').GetValues()[0,2]]
        
        #Initialise empty array for SJC experimental data
        EJCmrkrExp = np.empty([nFramesExp,3])
        
        #Loop through experimental trial frames and calculate SJC position
        for frameNo in range(0,nFramesExp):
            
            #Define the marker positions for below solver equations
            xA = AmrkrExp[frameNo,0]; yA = AmrkrExp[frameNo,1]; zA = AmrkrExp[frameNo,2];
            xB = BmrkrExp[frameNo,0]; yB = BmrkrExp[frameNo,1]; zB = BmrkrExp[frameNo,2];
            xC = CmrkrExp[frameNo,0]; yC = CmrkrExp[frameNo,1]; zC = CmrkrExp[frameNo,2];
            
            #Define a function that expresses the equations
            def f(p):
                x,y,z = p
                #Define functions
                fA = ((x - xA)**2 + (y - yA)**2 + (z - zA)**2) - distA_avg**2
                fB = ((x - xB)**2 + (y - yB)**2 + (z - zB)**2) - distB_avg**2
                fC = ((x - xC)**2 + (y - yC)**2 + (z - zC)**2) - distC_avg**2
                return [fA,fB,fC]
            
            #Test function with values
            #print(f([30,40,50]))
            
            #Check whether to update initial guess
            if frameNo != 0:
                #Update initial guess
                initialGuess = [EJCmrkrExp[frameNo-1,0],EJCmrkrExp[frameNo-1,1],EJCmrkrExp[frameNo-1,2]]
                
            #Solve equation for current marker position
            EJCmrkrExp[frameNo,0],EJCmrkrExp[frameNo,1],EJCmrkrExp[frameNo,2] = fsolve(f,initialGuess)
            
            #Cleanup
            #del(xA,xB,xC,yA,yB,yC,zA,zB,zC)
        
        #Add new marker back into c3d data as R.EJC
            
        #Create new empty point
        newPoint2 = btk.btkPoint(acqExp.GetPointFrameNumber())
        #Set label on new point
        newPoint2.SetLabel('R.EJC')
        #Set values for new marker from those calculated
        newPoint2.SetValues(EJCmrkrExp)
        #Append new point to acquisition object
        acqExp.AppendPoint(newPoint2)
    
    #Write new c3d file
    writerExp = btk.btkAcquisitionFileWriter()
    writerExp.SetInput(acqExp)
    writerExp.SetFilename(trialC3D[0:-4] + '_withJointCentres.c3d')
    writerExp.Update()
    
    return('Functional joint centres added')

# %% Function to convert c3d to trc file using btk module

def c3d2trc(fullFile, markerList, filtFreq = None):
    
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
    
    return('C3D file successfully converted to TRC')

# %% Function for adding events to different trial types
    
def addEvents(trialC3D):
    
    #Read in the C3D file
    readerC3D = btk.btkAcquisitionFileReader()        
    #Load in experimental c3d trial
    readerC3D.SetFilename(trialC3D)       
    #Update reader
    readerC3D.Update()
    #Get the btk acquisition object
    acqC3D = readerC3D.GetOutput() 
    
    if 'UpwardReach' in trialC3D:

        #Use the horizontal velocity of the hand marker to see where it starts
        #moving forward and backwards (X-axis) to identify movement segments
        
        #Get the X-axis hand marker data
        xHand = acqC3D.GetPoint('R.HAND').GetValues()[:,0]
        #Calculate the difference between subsequent marker positions
        xHandDiff = np.diff(xHand)
        
        #Find the first element with a negative value (skipping over the first
        #few frames to avoid any movements at the beginning). This will indicate
        #the index where the weight has been picked off the shelf. This also 
        #uses a threshold of 1 to avoid any small movements
        liftOff1 = np.where(xHandDiff[100:-1] < -1)[0][0] + 100
        #The next point where positive data occurs will be the reach to put the
        #weight back on the shelf
        putBack1 = np.where(xHandDiff[liftOff1:-1] > 1)[0][0] + liftOff1
        #The next negative point after this will be the point where the weight 
        #has been placed back on the shelf
        liftOff2  = np.where(xHandDiff[putBack1:-1] < -1)[0][0] + putBack1
        
        #Add the events back into the C3D file with relevant labels
        
        #Get the start frame to add to the identified row index
        firstFrame = acqC3D.GetFirstFrame()
        #Get sampling rate to calculate time
        fs = acqC3D.GetPointFrequency()
        
        #Concentric start point
        conStart = btk.btkEvent(); 
        conStart.SetFrame(int(putBack1) + firstFrame-1); conStart.SetLabel('conStart')
        conStart.SetTime((putBack1 + firstFrame-1) * (1/fs))
        acqC3D.AppendEvent(conStart)
        #Concentric end point
        conEnd = btk.btkEvent(); 
        conEnd.SetFrame(int(liftOff2) + firstFrame-1); conEnd.SetLabel('conEnd')
        conEnd.SetTime((liftOff2 + firstFrame-1) * (1/fs))
        acqC3D.AppendEvent(conEnd)
        #Eccentric start point
        eccStart = btk.btkEvent(); 
        eccStart.SetFrame(int(liftOff1) + firstFrame-1); eccStart.SetLabel('eccStart')
        eccStart.SetTime((liftOff1 + firstFrame-1) * (1/fs))
        acqC3D.AppendEvent(eccStart)
        #Eccentric end point
        eccEnd = btk.btkEvent(); 
        eccEnd.SetFrame(int(putBack1-1) + firstFrame-1); eccEnd.SetLabel('eccEnd')
        eccEnd.SetTime((putBack1-1 + firstFrame-1) * (1/fs))
        acqC3D.AppendEvent(eccEnd)
        #Update acquisition object
        acqC3D.Update()
        
    #Re-write C3D file
    writerC3D = btk.btkAcquisitionFileWriter()
    writerC3D.SetInput(acqC3D)
    writerC3D.SetFilename(trialC3D)
    writerC3D.Update()
    
    return('Events added to C3D file')
