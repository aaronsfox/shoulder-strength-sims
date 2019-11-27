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
import btk
import opensim as osim
import numpy as np
os.chdir('Code\Supplementary')
import osimHelperFunctions as osimHelper
import c3dHelperFunctions as c3dHelper
os.chdir('..\..')

# %% Scale model

#Navigate to data directory
os.chdir('ExpData')

#Add the functional joint centres to the static calibration trial ('Cal_Horizontal01.c3d')
c3dHelper.addFunctionalJointCentres('Cal_Horizontal01.c3d', shoulderC3D = 'ExtremeArc01.c3d', elbowC3D = 'ElbowFlex01.c3d', filtFreq = 6)

#Scale model from calibration trial

#Convert static trial to trc
staticC3D = os.getcwd() + '\Cal_Horizontal01_withJointCentres.c3d'
staticMarkerList = ['R.ACR','L.ACR','R.SJC','MCLAV','C7','T10','R.PUA.Top',
              'R.PUA.Left','R.PUA.Right','R.LUA.Top','R.LUA.Left',
              'R.LUA.Right','R.LELB','R.MELB','R.EJC','R.MWRI','R.LWRI',
              'R.FA.Top','R.FA.Left','R.FA.Right','R.FIN5','R.FIN1','R.HAND']
c3dHelper.c3d2trc(staticC3D,staticMarkerList)

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
trialList = list(['UpwardReach9001','UpwardReach9002'])

##### TO DO: Loop the IK across the different experimental data files...

for t in range(0,len(trialList)):
    
    #Set current trial name
    trialName = trialList[t]
    
    #Add events to the original c3d file
    c3dHelper.addEvents(trialName + '.c3d')
    
    #Add the functional joint centres to the current trial
    c3dHelper.addFunctionalJointCentres(trialName + '.c3d', shoulderC3D = 'ExtremeArc01.c3d', filtFreq = 6)
    
    #Convert current trial C3D to TRC
    trialC3D = os.getcwd() + '\\' + trialName + '_withJointCentres.c3d'
    dynamicMarkerList = ['R.ACR','R.SJC','MCLAV','C7','T10','R.PUA.Top',
                  'R.PUA.Left','R.PUA.Right','R.LUA.Top','R.LUA.Left',
                  'R.LUA.Right','R.MWRI','R.LWRI','R.FA.Top',
                  'R.FA.Left','R.FA.Right']
    c3dHelper.c3d2trc(trialC3D,dynamicMarkerList, filtFreq = 6)
    
    #General inputs for IK
    markerFile = os.getcwd() + '\\' + trialName + '_withJointCentres.trc'
    if not os.path.isdir(trialName):
        os.mkdir(trialName)     #create new directory for trial results
    outputDir = os.getcwd() + '\\' + trialName
    ikTaskSet = os.path.split(modelFile)[0] + '\Deakin_UpperLimb_RightSideOnly_IKTaskSet.xml'
    
    #Shift to output directory for current trial
    os.chdir(outputDir)
    
    #Identify time ranges for trials
    if 'UpwardReach' in trialName:
        
        #Create and run separate IK files for the concetric and eccentric parts        
        #Get the start and end times for each movement part from the c3d file
        
        #Get the c3d data
        c3dData = btk.btkAcquisitionFileReader()
        c3dData.SetFilename(trialC3D)
        c3dData.Update()
        c3dAcq = c3dData.GetOutput() 
        
        #Get the events data labels and times
        eventLabels = [None] * c3dAcq.GetEventNumber()
        eventTimes = {}
        for eventNo in range(0,c3dAcq.GetEventNumber()):
            eventLabels[eventNo] = c3dAcq.GetEvent(eventNo).GetLabel()
            eventTimes[eventLabels[eventNo]] = c3dAcq.GetEvent(eventNo).GetTime()

        #Run IK on the separate concentric and eccentric parts
        for n in range(0,2):
            if n == 0:
                #Set the time range
                timeRange = eventTimes['conStart'],eventTimes['conEnd']
                #Set the trial name
                trialNameIK = trialName + '_Concentric'
                #Generate and run IK tool
                osimHelper.setup_IKTool(modelFile = modelFile, markerFile = markerFile,
                    outputDir = outputDir, ikTaskSet = ikTaskSet,
                    trialName = trialNameIK, timeRange = timeRange)
                #Rename generic IK outputs
                os.rename('_ik_marker_errors.sto',trialNameIK + '_ik_marker_errors.sto')
                os.rename('_ik_model_marker_locations.sto',trialNameIK + '_ik_model_marker_locations.sto')
                #Cleanup
                del(timeRange,trialNameIK)
            else:
                #Set the time range
                timeRange = eventTimes['eccStart'],eventTimes['eccEnd']
                #Set the trial name
                trialNameIK = trialName + '_Eccentric'
                #Generate and run IK tool
                osimHelper.setup_IKTool(modelFile = modelFile, markerFile = markerFile,
                    outputDir = outputDir, ikTaskSet = ikTaskSet,
                    trialName = trialNameIK, timeRange = timeRange)
                #Rename generic IK outputs
                os.rename('_ik_marker_errors.sto',trialNameIK + '_ik_marker_errors.sto')
                os.rename('_ik_model_marker_locations.sto',trialNameIK + '_ik_model_marker_locations.sto')
                #Cleanup
                del(timeRange,trialNameIK)
                
        #Return to experimental data directory
        os.chdir('..')
        
        #Cleanup
        del(eventLabels,eventTimes,eventNo)
        
    ##### TO DO: add other elseif statements for other trial types
            
# %% Run inverse muscle drive solutions using MoCo

##### Test a MoCo states tracking solution with shoulder kinematics
##### Using generic Wu model

#Load the desired model
osimModel = osim.Model('X:\ShoulderInstability\MuscleStrengtheningSimulation\Models\BaselineModel.osim')

#Lock the thorax joints of the model to make this a shoulder only movement
coordSet = osimModel.updCoordinateSet()
coordSet.get('thorax_tilt').set_locked(True)
coordSet.get('thorax_list').set_locked(True)
coordSet.get('thorax_rotation').set_locked(True)
coordSet.get('thorax_tx').set_locked(True)
coordSet.get('thorax_ty').set_locked(True)
coordSet.get('thorax_tz').set_locked(True)

#Add torque actuators to the 
for c in range(0,coordSet.getSize()):
    if 'elbow_flexion' in coordSet.get(c).getName():
        #Add an idealised torque actuator (optimal force = 150)
        osimHelper.addCoordinateActuator(osimModel, coordSet.get(c).getName(), 150)
    elif 'pro_sup' in coordSet.get(c).getName():
        #Add an idealised torque actuator (optimal force = 75)
        osimHelper.addCoordinateActuator(osimModel, coordSet.get(c).getName(), 75)
    elif 'elv_angle' in coordSet.get(c).getName() \
    or 'shoulder_elv' in coordSet.get(c).getName() \
    or 'shoulder_rot' in  coordSet.get(c).getName():
        #Add low level reserve actuator (optimal force = 1)
        osimHelper.addCoordinateActuator(osimModel, coordSet.get(c).getName(), 1)

#Replace the muscles in the model with muscles from DeGroote, Fregly, 
#et al. 2016, "Evaluation of Direct Collocation Optimal Control Problem 
#Formulations for Solving the Muscle Redundancy Problem". These muscles
#have the same properties as the original muscles but their characteristic
#curves are optimized for direct collocation (i.e. no discontinuities, 
#twice differentiable, etc).
osim.DeGrooteFregly2016Muscle().replaceMuscles(osimModel)

#Turn off muscle-tendon dynamics to keep the problem simple.
#This is probably already done in the model anyway
for m in range(0,osimModel.getMuscles().getSize()):
    osimModel.updMuscles().get(m).set_ignore_tendon_compliance(True)
    
#Create moco tool
moco = osim.MocoTool();

#Configure the solver.
solver = moco.initCasADiSolver();
solver.set_num_mesh_points(25);
solver.set_dynamics_mode('implicit');
solver.set_optim_convergence_tolerance(1e-4);
solver.set_optim_constraint_tolerance(1e-4);
solver.set_optim_solver('ipopt');
solver.set_transcription_scheme('hermite-simpson');
solver.set_enforce_constraint_derivatives(True);
solver.set_optim_hessian_approximation('limited-memory');
solver.set_optim_finite_difference_scheme('forward');
    
#Set bounds on the problem.
problem = moco.updProblem();

#Set the time bounds (from kinematics file)
problem.setTimeBounds(osim.MocoInitialBounds(3.572), osim.MocoFinalBounds(6.044))

#Set kinematic bounds (mins and maxes)

#Shoulder elevation
charValue = '/jointset/' + coordSet.get('shoulder_elv').getJoint().getName() + '/' + coordSet.get('shoulder_elv').getName() + '/value';
charSpeed = '/jointset/' + coordSet.get('shoulder_elv').getJoint().getName() + '/' + coordSet.get('shoulder_elv').getName() + '/speed';
mn = coordSet.get('shoulder_elv').getRangeMin();
mx = coordSet.get('shoulder_elv').getRangeMax();
problem.setStateInfo(charValue,osim.MocoBounds(mn,mx))
problem.setStateInfo(charSpeed,osim.MocoBounds(-50,50));
del(mn,mx,charValue,charSpeed)

#Shoulder rotation
charValue = '/jointset/' + coordSet.get('shoulder_rot').getJoint().getName() + '/' + coordSet.get('shoulder_rot').getName() + '/value';
charSpeed = '/jointset/' + coordSet.get('shoulder_rot').getJoint().getName() + '/' + coordSet.get('shoulder_rot').getName() + '/speed';
mn = coordSet.get('shoulder_rot').getRangeMin();
mx = coordSet.get('shoulder_rot').getRangeMax();
problem.setStateInfo(charValue,osim.MocoBounds(mn,mx))
problem.setStateInfo(charSpeed,osim.MocoBounds(-50,50));
del(mn,mx,charValue,charSpeed)

#Elevation plane
charValue = '/jointset/' + coordSet.get('elv_angle').getJoint().getName() + '/' + coordSet.get('elv_angle').getName() + '/value';
charSpeed = '/jointset/' + coordSet.get('elv_angle').getJoint().getName() + '/' + coordSet.get('elv_angle').getName() + '/speed';
mn = coordSet.get('elv_angle').getRangeMin();
mx = coordSet.get('elv_angle').getRangeMax();
problem.setStateInfo(charValue,osim.MocoBounds(mn,mx))
problem.setStateInfo(charSpeed,osim.MocoBounds(-50,50));
del(mn,mx,charValue,charSpeed)

#Elbow angle
charValue = '/jointset/' + coordSet.get('elbow_flexion').getJoint().getName() + '/' + coordSet.get('elbow_flexion').getName() + '/value';
charSpeed = '/jointset/' + coordSet.get('elbow_flexion').getJoint().getName() + '/' + coordSet.get('elbow_flexion').getName() + '/speed';
mn = coordSet.get('elbow_flexion').getRangeMin();
mx = coordSet.get('elbow_flexion').getRangeMax();
problem.setStateInfo(charValue,osim.MocoBounds(mn,mx))
problem.setStateInfo(charSpeed,osim.MocoBounds(-50,50));
del(mn,mx,charValue,charSpeed)

#Forearm
charValue = '/jointset/' + coordSet.get('pro_sup').getJoint().getName() + '/' + coordSet.get('pro_sup').getName() + '/value';
charSpeed = '/jointset/' + coordSet.get('pro_sup').getJoint().getName() + '/' + coordSet.get('pro_sup').getName() + '/speed';
mn = coordSet.get('pro_sup').getRangeMin();
mx = coordSet.get('pro_sup').getRangeMax();
problem.setStateInfo(charValue,osim.MocoBounds(mn,mx))
problem.setStateInfo(charSpeed,osim.MocoBounds(-50,50));
del(mn,mx,charValue,charSpeed)

#Set the muscle activation state bounds
#Set initial activation to be zero
for m in range(0,osimModel.getMuscles().getSize()):
    #Get current muscle name
    muscName = osimModel.updMuscles().get(m).getName();
    #Create state string for setting state info
    stateStr = '/forceset/' + muscName + '/activation';
    #Set activation bounds. Include a specification that initial activation
    #must be zero for the majority of movements, with the exception of the
    #abduction 90-180 and abduction/external rotation 0-90 movements - as
    #these do not start in a neutral position
    problem.setStateInfo(stateStr,osim.MocoBounds(0,1))
    #Cleanup
    del(muscName,stateStr)
del(m)

#Update problem
problem = moco.updProblem();

#Finalise model connections
osimModel.finalizeConnections();

#Set model to the moco problem
problem.setModelCopy(osimModel);

#Set cost functions for task

#States tracking problem for the shoulder kinematics
statesCost = osim.MocoStateTrackingCost('statesCost',5)
#theoretically could load the relevant data into a time series table
statesCost.setReferenceFile('X:\ShoulderInstability\MuscleStrengtheningSimulation\shoulder-strength-sims\ExpData\UpwardReach9001\MR_JustShoulderKinematicStates.sto')

#Add the end point cost with a matched effort cost
problem.addCost(statesCost)
problem.addCost(osim.MocoControlCost('effort',1));

#Update the solver

#Update the underlying MocoCasADiSolver with the new problem.
solver = osim.MocoCasADiSolver.safeDownCast(moco.updSolver());
solver.resetProblem(problem);

#Set guess
solver.setGuess('bounds');

#Run predictive problem
solution = moco.solve();

### states based tracking took 56 minutes but generated OK looking muscle 
### activations for the movement - given only 25 mesh points were used they 
### were a little 'spiky' in parts, but this wasn't over the top...


# %%












