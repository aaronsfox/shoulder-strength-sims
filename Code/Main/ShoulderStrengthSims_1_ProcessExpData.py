# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 09:45:37 2019

@author: aafox

Script processes the experimental c3d files from lab data collection through
inverse kinematics. This script should be run from the main directory (i.e.
the path that includes the experimental data folder)

"""

#Import modules
import os
import opensim as osim
import numpy as np
os.chdir('Code\Supplementary')
import osimHelperFunctions as osimHelper
os.chdir('..\..')
import btk

# %% Scale model

#Navigate to data directory
os.chdir('ExpData')

#Add the functional joint centres to the static calibration trial ('Cal_Horizontal01.c3d')
osimHelper.addFunctionalJointCentres('Cal_Horizontal01.c3d', shoulderC3D = 'Circumduction01.c3d', elbowC3D = 'ElbowFlex01.c3d', filtFreq = 6)

#Scale model from calibration trial

#Convert static trial to trc
staticC3D = os.getcwd() + '\Cal_Horizontal01_withJointCentres.c3d'
staticMarkerList = ['R.ACR','L.ACR','R.SJC','MCLAV','C7','T10','R.PUA.Top',
              'R.PUA.Left','R.PUA.Right','R.LUA.Top','R.LUA.Left',
              'R.LUA.Right','R.LELB','R.MELB','R.EJC','R.MWRI','R.LWRI',
              'R.FA.Top','R.FA.Left','R.FA.Right','R.FIN5','R.FIN1','R.HAND']
osimHelper.btk_c3d2trc(staticC3D,staticMarkerList)

#Scaling inputs
os.chdir('..\ModelFiles')
genModFile = os.getcwd() + '\Deakin_UpperLimb_RightSideOnly.osim'
markerFile = staticC3D[0:-4] + '.trc'
measurementSetFile = os.getcwd() + '\Deakin_UpperLimb_RightSideOnly_Scale_MeasurementSet.xml'
markerPlacerFile = os.getcwd() + '\Deakin_UpperLimb_RightSideOnly_Scale_MarkerPlacer.xml'
modelName = 'MR'

#Generate and run scale tool
osimHelper.setup_ScaleTool(genericModelFile = genModFile, markerFile = markerFile,
                    outputDir = os.getcwd(), modelName = modelName, participantMass = 30,
                    preserveMass = True, measurementSetFile = measurementSetFile,
                    markerPlacerFile = markerPlacerFile, printToFile = False)

#Set variable to scaled model file
modelFile = os.getcwd() + '\\' + modelName + '_scaledAdjusted.osim'

"""

Need to figure out how to allocate mass based on whole body mass to model

Also need to scale muscle strength here


"""

# %% Run inverse kinematics

#Go to experimental data directory
os.chdir('..\ExpData')

#Create list of trial names
trialList = list(['UpwardReach9002'])

##### TO DO: Loop the IK across the different experimental data files...

for t in range(0,len(trialList)):
    
    #Set current trial name
    trialName = trialList[t]
    
    #Add the functional joint centres to the current trial
    osimHelper.addFunctionalJointCentres(trialName + '.c3d', shoulderC3D = 'Circumduction01.c3d', filtFreq = 6)
    
    #Convert current trial C3D to TRC
    trialC3D = os.getcwd() + '\\' + trialName + '_withJointCentres.c3d'
    dynamicMarkerList = ['R.ACR','R.SJC','MCLAV','C7','T10','R.PUA.Top',
                  'R.PUA.Left','R.PUA.Right','R.LUA.Top','R.LUA.Left',
                  'R.LUA.Right','R.MWRI','R.LWRI','R.FA.Top',
                  'R.FA.Left','R.FA.Right','R.HAND']
    osimHelper.btk_c3d2trc(trialC3D,dynamicMarkerList, filtFreq = 6)
    
    #General inputs for IK
    markerFile = os.getcwd() + '\\' + trialName + '_withJointCentres.trc'
    if not os.path.isdir(trialName):
        os.mkdir(trialName)     #create new directory for trial results
    outputDir = os.getcwd() + '\\' + trialName
    ikTaskSet = os.path.split(modelFile)[0] + '\Deakin_UpperLimb_RightSideOnly_IKTaskSet.xml'
    
    #Identify time ranges for trials
    if 'UpwardReach' in trialName:
        
        #Use the horizontal velocity of the hand marker to see where it starts
        #moving forward and backwards to identify movement segments
        #Read in the trc file
        readerTRC = btk.btkAcquisitionFileReader()        
        #Load in experimental c3d trial
        readerTRC.SetFilename(os.getcwd() + '\\' + trialName + '_withJointCentres.trc')       
        #Update reader
        readerTRC.Update()
        #Get the btk acquisition object
        acqTRC = readerTRC.GetOutput() 
        #Get the X-axis hand marker data
        xHand = acqTRC.GetPoint('R.HAND').GetValues()[:,0]
        #Calculate the difference between subsequent marker positions
        xHandDiff = np.diff(xHand)
        #Find the first element with a negative value (skipping over the first
        #few frames to avoid any movements at the beginning). This will indicate
        #the index where the weight has been picked off the shelf. This also 
        #uses a threshold of -0.05 to avoid any small movements
        liftOff1 = np.where(xHandDiff[100:-1] < -0.05)[0][0] + 100
        #The next point where positive data occurs will be the reach to put the
        #weight back on the shelf
        putBack1 = np.where(xHandDiff[liftOff1:-1] > 0.05)[0][0] + liftOff1
        #The next negative point after this will be the point where the weight 
        #has been placed back on the shelf
        liftOff2  = np.where(xHandDiff[putBack1:-1] < -0.05)[0][0] + putBack1
        #Get the times of the relevant events from the TRC file for IK
        #Create a time vector based on frame numbers and sample rate
        ff = acqTRC.GetFirstFrame(); lf = acqTRC.GetLastFrame(); fs = acqTRC.GetPointFrequency()
        time = np.arange(ff * (1/fs), lf * (1/fs), 1/fs)
        #Get start and end times for different components of the movement
        #First component will be the concentric, second the eccentric
        conStart = time[putBack1]; conEnd = time[liftOff2]
        eccStart = time[liftOff1]; eccEnd = time[putBack1-1]
        #Cleanup
        del(xHand,xHandDiff,liftOff1,liftOff2,putBack1,ff,time)
        
        #Shift to output directory for current trial
        os.chdir(outputDir)
        
        #Create and run separate IK files for the various start and end times for the task
        for n in range(0,2):
            if n == 0:
                #Set the time range
                timeRange = conStart,conEnd
                #Set the trial name
                trialNameIK = trialName + '_Concentric'
                #Generate and run IK tool
                osimHelper.setup_IKTool(modelFile = modelFile, markerFile = markerFile,
                    outputDir = outputDir, ikTaskSet = ikTaskSet,
                    trialName = trialNameIK, timeRange = timeRange)
                
                ##### TO DO: write setup for eccentric
                ##### TO DO: the IK function just doesn't seem to run here???
            
    
    
    
    
    ##### TO DO: split into sections based on events
    
    timeRange = None
    
    
    #Generate and run scale tool
    osimHelper.setup_IKTool(modelFile = modelFile, markerFile = markerFile,
                    outputDir = outputDir, ikTaskSet = ikTaskSet,
                    trialName = trialName)












