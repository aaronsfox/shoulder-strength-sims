# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 13:20:45 2019

@author: aafox

Set of functions that can be used to assist in processing data within OpenSim

"""

#Import necessary modules
import btk
import opensim as osim
import numpy as np
import lxml.etree as ET
import subprocess
   
# %% Function to set-up and run a scale tool
    
def setup_ScaleTool(genericModelFile = None, markerFile = None, outputDir = None,
                    modelName = 'ScaledModel', participantMass = -1, preserveMass = True,
                    measurementSetFile = None, markerPlacerFile = None,
                    printToFile = False):
    
    ####### TO DO: ADD PROMPT DIALOGS IF FILES AREN'T PRESENT (I.E. STILL NONE)...
    
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
    #Read in the trc file
    reader = btk.btkAcquisitionFileReader()        
    #Load in experimental c3d trial
    reader.SetFilename(markerFile)       
    #Update reader
    reader.Update()
    #Get the btk acquisition object
    acq = reader.GetOutput() 
    #Get the first and last frame numbers
    ff = acq.GetFirstFrame(); lf = acq.GetLastFrame()
    #Get sampling frequency
    fs = acq.GetPointFrequency()
    #Calcuate start and end time based on sampling frequency and frames
    startTime = ff * (1/fs); endTime = (lf-1) * (1/fs); 
    #Set time array
    timeArray = osim.ArrayDouble()
    timeArray.set(0,startTime); timeArray.set(1,endTime)
    #Set scale time
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
    
    return('Scaling completed')
    
# %% Function to run inverse kinematics tool

def setup_IKTool(modelFile = None, markerFile = None, outputDir = None,
                    ikTaskSet = None, timeRange = None,
                    trialName = 'movementTrial', constraintWeight = 20,
                    accuracy = 1e-05, reportErrors = True,
                    reportMarkerLocations = True):
    
    ####### TO DO: ADD PROMPT DIALOGS IF FILES AREN'T PRESENT (I.E. STILL NONE)...
    
    #Initialise an inverse kinematics tool
    ik = osim.InverseKinematicsTool()
    
    #Set IK task set
    taskFile = osim.OpenSimObject.makeObjectFromFile(ikTaskSet)
    ikTaskSet = osim.IKTaskSet().safeDownCast(taskFile)
    for nTask in range(0,ikTaskSet.getSize()-1):
        #Add task
        ik.getIKTaskSet().set(nTask,ikTaskSet.get(nTask))
    
    #Set the marker file
    ik.setMarkerDataFileName(markerFile)
    
    #Set time range to process data from trc file
    #Use the full files time range if nothing is provided
    if timeRange != None:
        #####TO DO: use the provided time range
        ik.setStartTime(timeRange[0]); ik.setEndTime(timeRange[1])
    else:
        #Grab the full time range from the trc file        
        #Read in the trc file
        reader = btk.btkAcquisitionFileReader()        
        #Load in experimental c3d trial
        reader.SetFilename(markerFile)       
        #Update reader
        reader.Update()
        #Get the btk acquisition object
        acq = reader.GetOutput() 
        #Get the first and last frame numbers
        ff = acq.GetFirstFrame(); lf = acq.GetLastFrame()
        #Get sampling frequency
        fs = acq.GetPointFrequency()
        #Calcuate start and end time based on sampling frequency and frames
        startTime = ff * (1/fs); endTime = (lf-1) * (1/fs); 
        #Set in IK tool
        ik.setStartTime(startTime); ik.setEndTime(endTime)
        
    #Set output motion file
    ik.setOutputMotionFileName(outputDir + '\\' + trialName + '_ik.mot')
    
    #Print setup tool to file for further editing
    ik.printToXML(outputDir + '\\' + trialName + '_SetupIK.xml')
    
    #Edit remaining properties (easier to do from XML)
    
    #Open XML setup file
    XMLtree = ET.parse(outputDir + '\\' + trialName + '_SetupIK.xml')
    XMLroot = XMLtree.getroot()

    #Set model
    XMLroot.find('InverseKinematicsTool').find('model_file').text = modelFile
    
    #Set constraint weight
    XMLroot.find('InverseKinematicsTool').find('constraint_weight').text = str(constraintWeight)
    
    #Set accuracy
    XMLroot.find('InverseKinematicsTool').find('accuracy').text = str(accuracy)
    
    #Set report errors
    if reportErrors:
        XMLroot.find('InverseKinematicsTool').find('report_errors').text = 'true'
    else:
        XMLroot.find('InverseKinematicsTool').find('report_errors').text = 'false'
    
    #Set report marker locations
    if reportMarkerLocations:
        XMLroot.find('InverseKinematicsTool').find('report_marker_locations').text = 'true'
    else:
        XMLroot.find('InverseKinematicsTool').find('report_marker_locations').text = 'false'
    
    #Re-write XML file
    XMLtree.write(outputDir + '\\' + trialName + '_SetupIK.xml')
    
    #Run IK tool
    
    #Setup program for running OpenSim XML
    def runProgram(argList):
        # arglist is like ['./printNumbers.sh']
            proc = subprocess.Popen(argList, 
                                    shell=False, bufsize=1, 
                                    stdout=subprocess.PIPE, 
                                    stderr=subprocess.STDOUT)
            while (True):
                # Read line from stdout, print, break if EOF reached
                line = proc.stdout.readline()
                line = line.decode()
                if (line == ""): break
                print line,
                
            rc = proc.poll() 
            print '\nReturn code: ', rc, '\n'
            return rc
    
    #Run tool
    cmdprog = 'opensim-cmd'; cmdtool = 'run-tool'; cmdfile = trialName + '_SetupIK.xml'
    cmdfull = [cmdprog, cmdtool, cmdfile]
    runProgram(cmdfull)
    
    return('Inverse kinematics completed')

# %% Function for adding coordinate actuator to a model object
    
def addCoordinateActuator(model, coordName, optForce):
    
    #Get coordinate set
    cs = model.updCoordinateSet()
    
    #Create actuator object
    actu = osim.CoordinateActuator()
    actu.setName('actu_' + coordName)
    actu.setCoordinate(cs.get(coordName));
    actu.setOptimalForce(optForce);
    actu.setMinControl(-1);
    actu.setMaxControl(1);
    
    #Add to model
    model.addComponent(actu)
    
# %%
    