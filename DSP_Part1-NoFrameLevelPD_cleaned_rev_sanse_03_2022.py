# this is a python script (.py) that:
#ANTON NAIM IBRAHIM - 9/3/2020; and Modified by Omar Nazih Omari - December 16th, 2020

#This code will be used to read RAW liver data (from the .rf , .yml, etc etc files that we get straight from Clarius) and convert them into numpy #arrays for the relevant data (rf, bmode, qus, nps) on a frame level, stored in both numpy arrays (separate .npy file for each frame) and in large #dataframes (each row has data for a relevant frame)

#################################################################################################################################

#! pip install tqdm
#! pip install joblib
#! pip install numba
#! pip install pyaml
#! pip install cloudstorage
#! pip install tensorflow

import numpy as np

import pandas as pd

import pickle

import json

from tqdm import tqdm, tqdm_notebook

import joblib
from joblib import Parallel, delayed
from joblib.externals.loky import set_loky_pickler
from joblib import parallel_backend
from joblib import Parallel, delayed
from joblib import wrap_non_picklable_objects

import multiprocessing

import numba as nb

import os

import sys

import time

import traceback

import yaml

from tensorflow.python.lib.io import file_io

import io
from io import BytesIO
from io import StringIO

import scipy.signal as sp

import cloudstorage as gcs
from google.cloud import storage

from IPython.display import clear_output

# MODIFIED TO WORK WITH GOOGLE CLOUD BUCKETS

# looks for videos that have rf.raw (don't need rf.yml)

# get all file names for a particular case ID
# gets the file name (really the video name) BEFORE the _rf.raw
# we DO NOT want anything from the _rf.raw.lzo files
def files_in_ID(ID):

    path = ID + '/extracted/'
    
    # get the blob iterator objects
    blob = bucket_pull.list_blobs(prefix=ID)
    # get relevant file names from blob objects
    fileList = [file.name for file in blob if ( file.name.endswith("_rf.raw") and file.name.startswith(path) )]
    new_list = np.asarray(fileList)
    # avoid duplicates
    new_list = np.unique(new_list)
    # trim the _rf.raw
    size = np.size(new_list)
    for ii in range(0,size):
        # lstrip and rstrip can cause rare errors, because they just look for the particular characters
        #new_list[ii] = new_list[ii].rstrip('_rf.raw').lstrip(path)
        strlen = len(new_list[ii])
        new_list[ii] = new_list[ii][27:strlen-7]
    
    new_list = list(new_list)
    
    return new_list


def info(ID, file):
    
    path = ID + '/extracted/'
    
    Files={}
    Files["directory"] = path
    Files["xmlName"] = "probes_usx.xml"
    #Files["ymlName"] = path + file + "_rf.yml"
    Files["ymlName"] = file + "_rf.yml"
    Files["name"] = file + "_rf.raw"
    imgInfo = read_CLariusinfo(Files["name"], Files["xmlName"], Files["ymlName"], Files["directory"])
    
    return imgInfo

def read_compute(ID,file,depth,width,ham,hanning,freq_num_for_att):
    
    # try a few times in case of weird one-off errors
    for ii in range(0,3):
        '''sms: glitch in buckets, first time you call it does not work (if successful break loop)'''
        try:
            imgInfo = info(ID,file)
            #print("a")
            # CENTER FREQUENCY IS 3 MHz
            # we have used 5MHz before, but 3MHz is preferred, and Now we're at 4MHz
            freq = 4#3
            bsc5=read_Clariusimg4(imgInfo, 2, freq,PhantomC,depth,width,ham,hanning,freq_num_for_att)
            #print("b")
            break
        except:
            pass
    
    return bsc5

def read_CLariusinfo(filename=None, xmlFilename=None,ymlFileName=None,  filepath=None):

    import numpy as np
    #from readprobe_CLarius import readprobe_CLarius
    try:
        import yaml
    except:
        print ('You do not have the YAML module installed.\n'+'Run: pip install pyaml to fix this' )
        quit()


    
    # Some Initilization
    studyID = filename[1:(len(filename) - 4)]
    studyEXT = filename[(len(filename) - 2):]

    rfFilePath=filepath+filename
    
    #print(rfFilePath)
    
    # get from google
    while True:
        try:
            temp = storage.blob.Blob(rfFilePath,bucket_pull)
            content = temp.download_as_string()
            break
        except:
            print('\t\tfile load error')
            pass
        
    # write to temp file, then use temp file
    tempFile1 = 'temp_files/tempFile1' + studyID[:27]
    fpoint = open(tempFile1, 'wb')
    fpoint.write(content)
    fpoint.close()

    # Open RF file for reading
    while True:
        try:
            with open(tempFile1, mode='r') as fid : 
                # load the header information into a structure and save under a separate file
                hinfo = np.fromfile(fid , dtype='uint32', count=5 )
            break
        except:
            print('\t\terror opening ' , tempFile1)
            pass
    
    # delete temp file
    try:
        os.remove(tempFile1)
    except OSError:
        pass
    
    #ymlFilePath = filepath+ymlFileName
    # get from google
    #temp = storage.blob.Blob(ymlFilePath,bucket_pull)
    #content = temp.download_as_string()
    # write to temp file, then use temp file
    #tempFile2 = 'temp_files/tempFile2' + studyID[:27]
    #fpoint = open(tempFile2, 'wb')
    #fpoint.write(content)
    #fpoint.close()
    
    # Load the yml file 
    #try:
         #Yml File of Clarius 
    #    with open(tempFile2, mode='r') as yml_fid :
    #        yml_data = yaml.load(yml_fid)
    #except yaml.YAMLError as exc:
    #    print("Error In Parsing : %s " % exc.__cause__)
    
     # delete temp file
    #try:
    #    os.remove(tempFile2)
    #except OSError:
    #    pass
    

    header = {'id': 0, 'frames': 0, 'lines': 0, 'samples': 0, 'sampleSize': 0}
    header["id"] = hinfo[0]
    header["nframes"] = hinfo[1] #frames
    header["w"] = hinfo[2] #lines
    header["h"] = hinfo[3] #samples
    header["ss"] = hinfo[4] #sampleSize

    #from yml file
    # transFreq=yml_data["transmit frequency"]
    # header["txf"] =transFreq[1:(len(transFreq)-3)] # transmit freq - also called center freq  
    header["txf"] =4 

    # sampling_rate=yml_data["sampling rate"]
    # header["sf"] = sampling_rate[1:(len(sampling_rate)-3)]  #sampling freq - also called receive freq = sampling rate
    header["sf"]=20000000
    '''Sergio: All these can be read from .yml file'''
    
    header["dr"] = 23 # Fixed from Usx Probe.xml file
    header["ld"] = 192 # lineDensity => num of lines is 192... standard. 

    info={}

    # For USX - must also read probe file for probe parameters probeStruct and the('probes.xml', header.probe);
    probeStruct = readprobe_CLarius(xmlFilename, 21)
    info["probeStruct"] = probeStruct
    # assignin('base','header', header)

    # Add final parameters to info struct
    info["studyMode"] ="RF"
    info["file"] = filename
    info["filepath"] = filepath
    info["probe"] = "clarius"
    info["system"] = "Clarius"
    info["studyID"] = studyID
    info["samples"] = header["h"]
    info["lines"] = header["w"]#probeStruct.numElements; % or is it > header.w; Oversampled line density?
    info["depthOffset"] = probeStruct["transmitoffset"]# unknown for USX
    info["depth"] = header["ss"] * 10 ** 1 #1275/8; % in mm; from SonixDataTool.m:603 - is it header.dr?
    info["width"] = header["dr"] * 10 ** 1 #1827/8; %info["probeStruct.pitch*1e-3*info["probeStruct.numElements; % in mm; pitch is distance between elements center to element center in micrometers
    info["rxFrequency"] = header["sf"]
    info["samplingFrequency"] = header["sf"]
    info["txFrequency"] = header["txf"]
    info["centerFrequency"] = header["txf"] #should be same as transmit freq - it's the freq. of transducer
    info["targetFOV"] = 0
    info["numFocalZones"] = 1#Hardcoded for now - should be readable
    info["numFrames"] = header["nframes"]
    info["frameSize"] = info["depth"] * info["width"]
    info["depthAxis"] = info["depth"]
    info["widthhAxis"] = info["width"]
    info["lineDensity"] = header["ld"]
    info["height"] = info["depth"]#This could be different if there is a target FOV
    info["pitch"] = probeStruct["pitch"]
    info["dynRange"] = 0# Not sure if available
    info["yOffset"] = 0
    info["vOffset"] = 0
    info["lowBandFreq"] = info["txFrequency"] - 0.5 * probeStruct["frequency"]["bandwidth"]
    info["upBandFreq"] = info["txFrequency"] + 0.5 * probeStruct["frequency"]["bandwidth"]
    info["gain"] = 0
    info["rxGain"] = 0
    info["userGain"] = 0
    info["txPower"] = 0
    info["power"] = 0
    info["PRF"] = 0

    # One of these is the preSC, the other is postSC resolutions
    info["yRes"] = ((info["samples"] / info["rxFrequency"] * 1540 / 2) / info["samples"]) * 10 ** 3#>> real resolution based on curvature
    info["yResRF"] = info["depth"] / info["samples"]#>> fake resolution - simulating linear probe
    info["xRes"] = (info["probeStruct"]["pitch"] * 1e-6 * info["probeStruct"]["numElements"] / info["lineDensity"]) * 10 ** 3 #>> real resolution based on curvature
    info["xResRF"] = info["width"] / info["lines"]#>> fake resolution - simulating linear probe


    # Quad 2 or accounting for change in line density 
    info["quad2X"]= 1

    # Ultrasonix specific - for scan conversion - from: sdk607/MATLAB/SonixDataTools/SonixDataTools.m:719
    info["Apitch"] = (info["samples"] / info["rxFrequency"]* 1540 / 2) / info["samples"]# Axial pitch - axial pitch - in metres as expected by scanconvert.m
    info["Lpitch"] = info["probeStruct"]["pitch"] * 1e-6 * info["probeStruct"]["numElements"] / info["lineDensity"]# Lateral pitch - lateral pitch - in meters
    info["Radius"] = info["probeStruct"]["radius"] * 1e-6
    info["PixelsPerMM"] = 8# Number used to interpolate number of pixels to be placed in a mm in image
    info["lateralRes"] = 1 / info["PixelsPerMM"]# Resolution of postSC
    info["axialRes"] = 1 / info["PixelsPerMM"]# Resolution of postSC

    #print ("Clarius Info : %s " % info)

    return info


def readprobe_CLarius(fileName=None, probeID=None):

    import numpy as np
    import xml.etree.ElementTree as et
    import xml.dom.minidom

    # get from google
    #temp = storage.blob.Blob(fileName,bucket_pull)
    #content = temp.download_as_string()
    # write to temp file, then use temp file
    #fpoint = open('temp_files/tempFile3', 'wb')
    #fpoint.write(content)
    #fpoint.close()

    # Open the probes.xml file and read it into mem
    #fid = open('temp_files/tempFile3', mode='r')
    fid = open('temp_files/xml_file', mode='r')
    if (fid is None):
        print("Could not find the probes.xml file ")
        Probe = {}
        return
   
    xml = xml.dom.minidom.parse(fid) # or xml.dom.minidom.parseString(xml_string)
    root=et.fromstring(xml.toprettyxml())
    # Picking out the text relating to the probe
    probeRoot=root.find("./probe[@id='%s']" % str(probeID))
    
    # create empty Dictionary for saving data by its keys 
    Probe = {}
    probeName=probeRoot.attrib["name"]
    Probe["name"]=probeName

    for elem in probeRoot.iter():
        # #print(elem.tag,elem.attrib,elem.text)
        if(elem.tag=="biopsy"):
            biopsy=elem.text
            #print("Biopsy Text %s " % biopsy)
            Probe["biopsy"] =biopsy
        elif(elem.tag=="vendors"):
            vendors=elem.text
            #print("Vendors Text %s " % vendors)
            Probe["vendors"] =vendors
        elif(elem.tag=="type"):
            probe_type=elem.text
            #print("Type Text %s " % probe_type)
            Probe["type"] =int(probe_type)
        elif(elem.tag=="transmitoffset"):
            transmitoffset=elem.text
            #print("transmitoffset Text %s " % transmitoffset)
            Probe["transmitoffset"] =int(float(transmitoffset))
        elif(elem.tag=="center"):
            freqCenter=elem.text
            #print("freCenter Text %s " % freqCenter)
        elif(elem.tag=="bandwidth"):
            freqBandwith=elem.text
            #print("freqBandwith Text %s " % freqBandwith)
        elif(elem.tag=="maxfocusdistance"):
            maxfocusdistance=elem.text
            #print("maxfocusdistance Text %s " % maxfocusdistance)
            Probe["maxfocusdistance"] =int(maxfocusdistance)
        elif(elem.tag=="maxfocusdistance"):
            maxfocusdistance=elem.text
            #print("maxfocusdistance Text %s " % maxfocusdistance)
            Probe["maxfocusdistance"] =int(maxfocusdistance)
        elif(elem.tag=="maxsteerangle"):
            maxsteerangle=elem.text
            #print("maxsteerangle Text %s " % maxsteerangle)
            Probe["maxsteerangle"] =int(maxsteerangle)
        elif(elem.tag=="minFocusDistanceDoppler"):
            minFocusDistanceDoppler=elem.text
            #print("minFocusDistanceDoppler Text %s " % minFocusDistanceDoppler)
            Probe["minFocusDistanceDoppler"] =int(minFocusDistanceDoppler)
        elif(elem.tag=="minlineduration"):
            minlineduration=elem.text
            #print("minlineduration Text %s " % minlineduration)
            Probe["minlineduration"] =int(minlineduration)
        elif(elem.tag=="FOV"):
            probMotorFOV=elem.text
            #print("probMotorFOV Text %s " % probMotorFOV)
        elif(elem.tag=="homeMethod"):
            probMotorHomeMethod=elem.text
            #print("probMotorHomeMethod Text %s " % probMotorHomeMethod)
        elif(elem.tag=="minTimeBetweenPulses"):
            probMotorminTimeBetweenPulses=elem.text
            #print("probMotorminTimeBetweenPulses Text %s " % probMotorminTimeBetweenPulses)
        # elif(elem.tag=="radius"):
        #     probMotorRadius=elem.text
        #     #print("probMotorRadius Text %s " % probMotorRadius)
        elif(elem.tag=="steps"):
            probMotorSteps=elem.text
            #print("probMotorSteps Text %s " % probMotorSteps)
        elif(elem.tag=="homeCorrection"):
            probMotorHomeCorrection=elem.text
            #print("probMotorHomeCorrection Text %s " % probMotorHomeCorrection)
        elif(elem.tag=="numElements"):
            numElements=elem.text
            #print("numElements Text %s " % numElements)
            Probe["numElements"] =int(numElements)
        elif(elem.tag=="pinOffset"):
            pinOffset=elem.text
            #print("pinOffset Text %s " % pinOffset)
            Probe["pinOffset"] =int(pinOffset)
        elif(elem.tag=="pitch"):
            pitch=elem.text
            #print("pitch Text %s " % pitch)
            Probe["pitch"] =int(pitch)
        elif(elem.tag=="radius"):
            radius=elem.text
            #print("radius Text %s " % radius)
            Probe["radius"] =int(radius)
        elif(elem.tag=="support"):
            support=elem.text
            #print("support Text %s " % support)
            Probe["support"] =support
        elif(elem.tag=="muxWrap"):
            muxWrap=elem.text
            #print("muxWrap Text %s " % muxWrap)
            Probe["muxWrap"] =muxWrap
        elif(elem.tag=="elevationLength"):
            elevationLength=elem.text
            #print("elevationLength Text %s " % elevationLength)
            Probe["elevationLength"] =int(float(elevationLength))
        elif(elem.tag=="maxPwPrp"):
            maxPwPrp=elem.text
            #print("maxPwPrp Text %s " % maxPwPrp)
            Probe["maxPwPrp"] =int(maxPwPrp)
        elif(elem.tag=="invertedElements"):
            invertedElements=elem.text
            #print("invertedElements Text %s " % invertedElements)
            Probe["invertedElements"] =int(invertedElements)
        

    Probe["frequency"]={"center":int(freqCenter),"bandwidth":int(freqBandwith)}
    Probe["motor"] = {"FOV":int(probMotorFOV),"homeMethod":int(probMotorHomeMethod),
                    "minTimeBetweenPulses":int(probMotorminTimeBetweenPulses),"motor_radius":int(0),
                    "steps":int(probMotorSteps),"homeCorrection":int(probMotorHomeCorrection), }
    #print("***************************************************************************")


    return Probe


def read_Clariusimg4(Info=None, frame=None,freq=None,phant=None,depth=None,width=None,ham=None,hanning=None,freq_num_for_att=None):
    '''sms: start of qus calculation'''
    #Modded by Adi to include QUS calculation
    #returns tree with spots = num frames
    #tree[n].rf=raw rf data
    #tree[n].bsb=tree holding qus data
    #tree[n].newtrial=2d array holding QUS data
    import os
    import sys

    #sys.path.append(os.path.dirname('./solution/clarius_read/'))
    #sys.path.append(os.path.dirname('./solution/convert/'))

    #from PRead_Clarius import PRead_Clarius
    #from PRead_Clarius2 import PRead_Clarius2
    #from rf2bmode import rf2bmode
    #from scanconvert_mapped import scanconvert_mapped
    #from scanconvert import scanconvert


    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Read ultrasonix rf data from tablet and touch. Needs meta data stored as struct.
    # The ModeIM struct contains a map of x and y coordiantes to retrace a
    # point in a scan conerted image back to the original non-scanconverted RF
    # data.
    #Input:
    # Info - meta data with parameters for image, analysis and display
    # frame - frame number to read
    #Output:
    # Bmode - Scan converted bmode image for display. 
    # ModeIM - Contains: .orig (original RF data for image), .data (scan converted RF data), .xmap (x coordinates of point on original data), .ymap (ycoordinate of point of original data)
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # Get RF data file path using the metadata in Info
    usx_rf_filepath = Info["filepath"]+Info["file"]

    # Read the image data
    ModeIM=PRead_Clarius(usx_rf_filepath, '6.0.3' )
    '''sms: ModeIM rf data (full-video)'''
    ar3=Tree()
    frame=frame-1
    DF = pd.DataFrame()
    #ar3 = pd.DataFrame()
    #Image calculation was removed for speed reasons. RF Data is preserved so it should be calculatable later
    #if you need it, uncomment the next 2 lines and the spot where scbmode is added to ar3
    if(np.shape(ModeIM)[0]==2928):
        '''sms: was frame cut in the long depth?'''
        #Bmode = rf2bmode(ModeIM)
        #print(np.shape(Bmode))
        #scBmode = np.array([(scanconvert(Bmode[:,:,m], Info)) for m in range(0,np.shape(ModeIM)[2])])
        for i in range(0,np.shape(ModeIM)[2]):
            # Make ModeIM just one frame - the chosen frame
            ModeIMi = ModeIM[:,:, i]
            scBmode=rf2bmode(ModeIMi)#(scanconvert(rf2bmode(ModeIMi), Info))
            '''sms: b-mode used for segmentation -> there is a later version not tested for segmentation yet'''
            '''sms: it can be found in Kfold_newprocessing'''
            '''sms: it is not clear if human-better B-mode works better for segmentation'''
            #pass the rf, phantom, and freq to function, get back tree and 2d array
            #(newtrial,bsb)=treeCreator3Clar8(ModeIMi,phant,freq)
            
            #(ps,nps) = NPSCreator3Clar8(ModeIMi,phant,freq)
            (ps,nps) = NPSCreator3Clar81(ModeIMi,phant,freq,depth,hanning,width)
            '''sms: nps is currently not used'''
            #(qus,quant)=treeCreator3Clar8(ModeIMi,phant,freq)
            (qus,quant)=treeCreator3Clar81(ModeIMi,phant,freq,depth,ham,width,freq_num_for_att)

            nps[np.isnan(nps)]=0
            nps[np.isinf(nps)]=1
            
            qus[np.isnan(qus)]=0
            qus[np.isinf(qus)]=1
            
            quant[np.isnan(quant)]=0
            quant[np.isinf(quant)]=1
            
            
            Data = {
                'rf': [ModeIMi],
                'nps': [nps],
                'qus': [qus],
                'Bmode':[scBmode],
                'quant': [quant],
                'info': [Info],
                'ps': [ps]
            }

            df = pd.DataFrame(Data, columns = ['rf',
                                               'nps',
                                               'qus',
                                               'Bmode',
                                               'quant',
                                               'info',
                                               'ps'
                                              ] 
                             ) 
            
            ############################################################################
            ############################################################################
            ############################################################################

            DF = pd.concat([DF,df], axis = 0)

    else:
        print("\t\tbad pixels")
        ar3=[]
    return DF


def PRead_Clarius(filename,version='6.0.3'): 
    
    # Some Initilization
    studyID = filename[1:(len(filename) - 4)]
    studyEXT = filename[(len(filename) - 2):]
    
    # get from google
    while True:
        try:
            temp = storage.blob.Blob(filename,bucket_pull)
            content = temp.download_as_string()
            break
        except:
            print('\t\tfile load error')
            pass
        
    # write to temp file, then use temp file
    tempFile4 = 'temp_files/tempFile4' + studyID[27:]
    fpoint = open(tempFile4, 'wb')
    fpoint.write(content)
    fpoint.close()
    
    while True:
        try:
            fid = open(tempFile4,  mode='rb')
            break
        except:
            print('\t\terror opening ' , tempFile4)
            pass
    

# read the header info
    hinfo = np.fromfile(fid , dtype='uint32', count=5 )
    header = {'id': 0, 'nframes': 0, 'w': 0, 'h': 0, 'ss': 0}
    header["id"] = hinfo[0]
    header["nframes"] = hinfo[1] #frames
    header["w"] = hinfo[2] #lines
    header["h"] = hinfo[3] #samples
    header["ss"] = hinfo[4] #sampleSize


# % ADDED BY AHMED EL KAFFAS - 22/09/2018
    frames = header["nframes"]

    id = header["id"]
    if(id==0):  #iq
        ts = np.zeros(shape=(1,frames))
        data = np.zeros(shape=(frames, header["h"]*2, header["w"]))
    #  read ENV data
        for f in range (frames): 
        # read time stamp
            ts[f] = np.fromfile(fid , dtype='uint64', count=1 )
        # read one line
            oneline = np.fromfile(fid, dtype='uint16').reshape((header["h"]*2, header["w"])).T
            data[f,:,:] = oneline
#######################################################################################################
    elif(id==1): #env
        ts = np.zeros(shape=(1,frames))
        data = np.zeros(shape=(frames, header["h"], header["w"])) 
    #  read ENV data
        for f in range (frames): 
        # read time stamp
            ts[f] = np.fromfile(fid , dtype='uint64', count=1 )
        # read one line
            oneline = np.fromfile(fid, dtype='uint8').reshape((header["h"], header["w"])).T
            data[f,:,:] = oneline
#######################################################################################################
    elif(id==2): #RF
        ts = np.zeros(shape=(1,frames))
        data = np.zeros(shape=(header["h"], header["w"],frames))
    #  read RF data
        for f in range (frames):    
            v = np.fromfile(fid, count=header["h"]*header["w"] , dtype='int16' )
            data[:,:,f] =np.flip(v.reshape(header["h"], header["w"],order ='F').astype(np.int16), axis=1)
#######################################################################################################
    elif(id==3):
        ts = np.zeros(shape=(1,frames))
        data = np.zeros(shape=(frames, header["h"]*2, header["w"])) 
    #  read  data
        for f in range (frames): 
            # read time stamp
            ts[f] = np.fromfile(fid , dtype='int64', count=1 )
            # read one line
            oneline = np.fromfile(fid, dtype='int16').reshape((header["h"]*2, header["w"])).T
            data[f,:,:] = oneline
    
    # delete temp file
    try:
        os.remove(tempFile4)
    except OSError:
        pass

    return data


def rf2bmode(rf):
    import numpy as np
    from scipy.signal import hilbert,decimate,resample


    # decimation factor use to create Env from RF
    decimationFactor = 1

    # make a compression table
    alpha = 0.55# sqrt compression, change for a different compression table
    denom = np.exp(alpha * np.log(65535)) / 255.0
    CompressionTable = np.exp(alpha * np.log(np.arange(0,65536))) / denom 
    # CompressionTable = np.exp(alpha * np.log(65535)) / denom
    
    # calculate envelope and log compress
    Env = 1 + np.fix(np.abs(hilbert(rf, axis= 0)))
    Env = CompressionTable[np.array(Env, dtype=int)]

    EnvDec =resample(Env ,int(len(Env[ : , 0])/decimationFactor))
    
    EnvDec[(EnvDec>255).nonzero()]=255
#EnvDec[(EnvDec>255).nonzero()]=255
    return EnvDec


def scanconvert(Iin=None, info=None):#Apitch, Lpitch, Radius, PixelsPerMM)
    import numpy as np
    if (info["Radius"] == 0):    # A linear array probe
        #print("Linear Array")
        AxialExtent = info["Apitch"] * np.size(Iin, 0)
        LateralExtent = info["Lpitch"] * np.size(Iin, 1)
        AxialSize = AxialExtent * 1000 * info["PixelsPerMM"]
        LateralSize = LateralExtent * 1000 * info["PixelsPerMM"]
        Iout = np.resize(Iin, (LateralSize, AxialSize))    # cubic interpolation
    else:    # A curved array probe    
        t = ((np.arange(1,Iin.shape[1],dtype=int)) - Iin.shape[1]/ 2) * info["Lpitch"] / info["Radius"]
        r = info["Radius"] + (np.arange(1,Iin.shape[0])) * info["Apitch"]
        [t, r] = np.meshgrid(t, r)
        x = np.multiply(r, np.cos(t))
        y = np.multiply(r, np.sin(t))
    
        divides=1e-3 / info["PixelsPerMM"]
        xMin=np.min(x)
        xMax=np.max(x)
        #print("X Min : %s , X Max %s" % (xMin,xMax))
        #print("Divides : %s " % divides)
        xreg =  np.arange(np.min(x) , np.max(x) ,divides)
        yreg =  np.arange(np.min(y) , np.max(y) ,divides)
        [yreg, xreg] = np.meshgrid(yreg, xreg) 
        Iout = np.zeros(xreg.shape)
        xCntr = np.arange(0,xreg.shape[1])
        Cntry = np.arange(0,yreg.shape[0])
        theta = np.array([cart4pol(xreg[yCntr, xCntr], yreg[yCntr, xCntr]) for yCntr in Cntry])
        rho = np.array([cart2pol(xreg[yCntr, xCntr], yreg[yCntr, xCntr]) for yCntr in Cntry])
        #for xCntr in tqdm(np.arange(0,xreg.shape[1])):
            #for yCntr in np.arange(0,yreg.shape[0]):
                #[theta, rho] = cart2pol(xreg[yCntr, xCntr], yreg[yCntr, xCntr])
                #print("theta : %s , rho %s" % (theta,rho))
        indt = np.floor(theta / (info["Lpitch"] / info["Radius"]) + Iin.shape[1] / 2) 
        indr = np.floor((rho - info["Radius"]) / info["Apitch"]) 

        [j,i]=((indt>=0)&(indt <= t.shape[1])& (indr>= 0) &(indr <= r.shape[0])).nonzero()
        #print(np.size(j))
        j=j.astype(int)
        i=i.astype(int)
        Iout[j, i] = Iin[indr[j,i].astype(int), indt[j,i].astype(int)]
        #for i in xCntr:
            
         #   for j in Cntry:
                
          #      if indt[j,i] >= 0 and (indt[j,i] <= t.shape[1]and (indr[j,i] >= 0 and indr[j,i] <= r.shape[0])):
                    
           #         Iout[int(Cntry[j]), int(xCntr[i])] = Iin[int(indr[j,i]), int(indt[j,i])]

    return Iout


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    return rho
def cart4pol(x, y):
    phi = np.arctan2(y, x)
    return phi
def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)


#@nb.njit(fastmath=True)
def BSCcalc2(phantom,rf,freq,p,sz):
    #p= points, as the liver is 3904 points deep and 15cm, depth=15*p/3904
    z=15*p/sz#3904
    #according to the documents I was given, bsc in phantom =10^-3
    #attenuation coefficient = 0.5
    #attenuation in liver is 0.59 based on https://onlinelibrary.wiley.com/doi/full/10.7863/jum.2002.21.7.783
    #BSC=4.5*10^-5*freq^1.6*e^(-4z)(0.3f^1.28-0.55f)*fft/fft
    ab=0.59
    #odd trial in idiocy I guess phantom1=phantom[:,0:int(np.floor(np.shape(phantom)[1]/2)),:]
    #rf1=phantom[:,0:int(np.floor(np.shape(rf)[1]/2)),:]
    rf1=rf[:,0:int(np.floor(np.shape(rf)[1]/2)),:]
    phantom1=phantom[:,0:int(np.floor(np.shape(rf)[1]/2)),:]
    #rf1=10*np.log10(rf1)#**2)
    #phantom1=10*np.log10(phantom1)#**2)
    freq=freq#*1000000
    #BSC1=np.exp(np.log(rf1/phantom1)+freq*4*z/8.686*(ab-0.5)+np.log(10**-3))#half a sec ago
    #BSC1=np.exp(np.log((rf1/phantom1))+np.log(10**-3))#kinda works
    
    #BSC1=10**(-3)*np.exp(-4*z*((0.5-ab)*freq))* (np.abs(rf)**2/np.abs(phantom)**2)
    #BSC1=10**(-3)*np.exp(-4*z*((0.5-ab)))* (np.abs(rf1)**2/np.abs(phantom1)**2)#10**(-3)*np.exp(-4*z*((0.5-ab)))* (np.abs(rf)**2/np.abs(phantom)**2)
    
    #BSC1=(10**(-3))*np.log10(np.abs(rf1)**2/np.abs(phantom1)**2)+np.log10((-z*freq*((0.5-ab)))/10)# No idea what the fuck is going on with this formulation
    
    #BSC1=(10**(-3))*20*np.log10(np.abs(rf1)**2/np.abs(phantom1)**2)+((-z*freq*((0.5-ab)))/10)#p.bad
    #BSC1=(10**(-3))*(np.abs(rf1)**2/np.abs(phantom1)**2)*10**((-z*((0.5-ab))/10))
    
    #BSC1=10**(-3)*np.exp(-4*z*(freq*(0.5-ab)/100000000))*(np.abs(rf1)**2/np.abs(phantom1)**2)#important one
    #mb=int(np.floor(np.shape(rf1)[1]/8))
    mba=int(np.floor(np.shape(rf1)[1]*0.325))
    mbb=int(np.floor(np.shape(rf1)[1]*0.75))
    
    rf1=rf1[:,mba:mbb,:]
    phantom1=phantom1[:,mba:mbb,:]
    
    BSC1=(10**(-3))*(np.abs(rf1)**2/np.abs(phantom1)**2)*10**(-4*z*freq*((0.5-ab)/(20)))
    
    #BSC1=(10**(-3))*(np.abs(rf1)**2/np.abs(phantom1)**2)*np.exp(-4*z*freq*((0.5-ab)/(8.686*10000)))
    
    #BSC1=(10**(-3))*(np.abs(rf1)**2/np.abs(phantom1)**2)*np.exp(-4*z*freq*((0.5-ab)/(8.686*100000)))#important one 
    return np.abs(BSC1)

#@nb.njit(fastmath=True)
def BSCcalc3(phantom,rf,freq,p,ab,sz):
    '''sanse: BSC Oelze book'''
    
    #same as above, but uses the calculated attenuation coefficient
    #p= points, as the liver is 3904 points deep and 15cm, depth=15*p/3904
    z=15*p/sz#3904
    #according to the documents I was given, bsc in phantom =10^-3
    #attenuation coefficient = 0.5
    #attenuation in liver is 0.59 based on https://onlinelibrary.wiley.com/doi/full/10.7863/jum.2002.21.7.783
    #BSC=4.5*10^-5*freq^1.6*e^(-4z)(0.3f^1.28-0.55f)*fft/fft
    #ab=0.59
    #phantom1=phantom[:,0:int(np.floor(np.shape(phantom)[1]/2)),:]
    #rf1=phantom[:,0:int(np.floor(np.shape(rf)[1]/2)),:]
    freq=freq#*1000000
    rf1=rf[:,0:int(np.floor(np.shape(rf)[1]/2)),:]
    phantom1=phantom[:,0:int(np.floor(np.shape(rf)[1]/2)),:]
    #rf1=10*np.log10(rf1)#**2)
    #phantom1=10*np.log10(phantom1)#**2)
    
    #BSC1=10**(-3)*np.exp(-4*z*((0.5-ab)))*(np.abs(rf)**2/np.abs(phantom)**2)
    #ab=0.9
    
    mb=int(np.floor(np.shape(rf1)[1]/8))
    mba=int(np.floor(np.shape(rf1)[1]*0.325))
    mbb=int(np.floor(np.shape(rf1)[1]*0.75))
    rf1=rf1[:,mba:mbb,:]
    phantom1=phantom1[:,mba:mbb,:]
    
    
    
    #BSC1=np.exp(np.log(rf1/phantom1)+freq*4*z/8.686*(ab-0.5)+np.log(10**-3))#half a sec ago
    #BSC1=np.exp(np.log((rf1/phantom1)*freq*4*z/8.686*(ab-0.5))+np.log(10**-3))#kinda works
    #BSC1=np.exp(np.log((rf1/phantom1))+np.log(10**-3))#kinda works
    #BSC1=(10**(-3))*np.log10(np.abs(rf1)**2/np.abs(phantom1)**2)+np.log10((-z*freq*((0.5-ab)))/10)# No idea what the fuck is going on with this formulation

    #BSC1=(10**(-3))*20*np.log10(np.abs(rf1)**2/np.abs(phantom1)**2)+((-z*freq*((0.5-ab)))/10)#p.bad
    #BSC1=(10**(-3))*(np.abs(rf1)**2/np.abs(phantom1)**2)*10**((-z*((0.5-ab))/10))
    
    #BSC1=10**(-3)*np.exp(-4*z*(freq*(0.5-ab)/100000000))*(np.abs(rf1)**2/np.abs(phantom1)**2)
    BSC1=(10**(-3))*(np.abs(rf1)**2/np.abs(phantom1)**2)*10**(-4*z*freq*((0.5-ab)/(20)))
    #BSC1=(10**(-3))*(np.abs(rf1)**2/np.abs(phantom1)**2)*np.exp(-4*z*freq*((0.5-ab)/(8.686*10000)))#  
    
    #BSC1=(10**(-3))*(np.abs(rf1)**2/np.abs(phantom1)**2)*np.exp(-4*z*freq*((0.5-ab)/(8.686*100000)))#important one 
    return np.abs(BSC1)

    
def NakaGamiParam2(c,v):
    '''sms: definition from Wikipedia'''
    from scipy.signal import hilbert
    #c is patient rf data, v is phantom rf data
    r=hilbert(c)
    p=hilbert(v)
    #w=np.nanmean((r)**2)
    #u=((np.nanmean((r)**2))**2)/(np.nanmean((((r)**2)-np.nanmean((r)**2))**2))
    w=np.nanmean((r/p)**2,axis=(1,2))
    u=((np.nanmean((r/p)**2,axis=(1,2)))**2)/(np.nanmean((((r/p)**2)-np.nanmean((r/p)**2,axis=(1,2)).reshape(-1,1,1))**2,axis=(1,2)))
    return[w,u]

def AttenuationCoeff42(phantom,rf,freq,p,sz,f,num):
    '''sms: amalgamation of ten different papers (mainly book of QUS in soft tissue: Oelze, Chapter on attenuation)
        it should be documented - notes in Adi's journal'''

    from scipy.signal import hilbert
    #p= points, as the liver is 3904 points deep and 15cm, depth=15*p/3904/start depth
    #choose size of proximal and distal windows as 25 points
    distal_wind = num#100#50#25
    zpx=15*p/sz#3904
    zds=15*(p+rf.shape[1]-distal_wind)/sz#3904
    #f=np.linspace(freq-2,freq+2,num=25)
#     zn=15*(rf.shape[1]-2*distal_wind)/2928#3904
#     zn=15*(rf.shape[1]-distal_wind)/2928#3904
    
    winsz=num
    
    #according to the documents I was given, bsc in phantom =10^-3
    ap=0.5
    #attenuation coefficient = 0.5 for phantom at 3 mhz 
    #as= attenuation coefficient of saline, approximate as water using: As=15.7*10^-17*f^2
    #bandwith=4 mhz?
    #q is the ratio of depth to aperture radius (or something), evidently 6 is a safe number to use?
    #if we ever use propietary devices, I want that!!
    #if so:
    ham=1#sp.hamming(distal_wind).reshape((distal_wind,1))###HERE
    #flat=sp.flattop(25).reshape((25,1))
    #tuk=sp.tukey(25).reshape((25,1))
    ham=1
    rfpx=hilbert(rf[:,0:distal_wind,:],axis=1)
    rfds=hilbert(rf[:,-distal_wind:,:],axis=1)
    phanpx=hilbert(phantom[:,0:distal_wind,:],axis=1)
    phands=hilbert(phantom[:,-distal_wind:,:],axis=1)
    

    ham=1
    rfPSHampx=np.fft.fft(rfpx*ham,axis=1,n=1*winsz)[:,:100,:]#)#we[l-36:l+36,:]
    rfPSHamds=np.fft.fft(rfds*ham,axis=1,n=1*winsz)[:,:100,:]
    
    pPSHampx=np.fft.fft(phanpx*ham,axis=1,n=1*winsz)[:,:100,:]#)#we[l-36:l+36,:]
    pPSHamds=np.fft.fft(phands*ham,axis=1,n=1*winsz)[:,:100,:]

#     SHamds=10*np.log(np.abs((rfPSHamds**2)/(pPSHamds**2)))
    SHampx=np.log(np.abs((rfPSHampx**2)/(pPSHampx**2)))###HERE
    SHamds=np.log(np.abs((rfPSHamds**2)/(pPSHamds**2)))#####HERE

    
    SHam=SHampx-SHamds
  
    
    mba=int(np.floor(np.shape(SHam)[1]*0.25))
    mbb=int(np.floor(np.shape(SHam)[1]*0.5))
    #SHam=SHam[mb*2:mb*5,:]
    SHam=SHam[:,mba:mbb,:]#[mba:mbb,:]


    f2=f.reshape((100,1))
    h=(ap*f2[mba:mbb,:])#**2

    qes=-(SHam/(4*(zds-zpx)))-h
    
    return qes#np.nanmean(np.abs(abham))#[np.nanmean(np.abs(abrec)), np.nanmean(np.abs(abham)), np.nanmean(np.abs(abflat)), np.nanmean(np.abs(abtuk))]

def ESDAC3(phantom,rf,freq,p,sz,a,b):
    '''sanse: ESD EAC - > based on a different paper (Adi will send)'''
    #p= points, as the liver is 3904 points deep and 15cm, depth=15*p/3904
    z=15*p/sz#3904
    '''sanse: depth we are acquiring at, may change'''
    #according to the documents I was given, bsc in phantom =10^-3
    #attenuation coefficient = 0.5 for phantom at 3 mhz 
    #as= attenuation coefficient of saline, approximate as water using: As=15.7*10^-17*f^2
    #bandwith=4 mhz?
    #q is the ratio of depth to aperture radius (or something), evidently 6 is a safe number to use?
    #if we ever use propietary devices, I want that!!
    #if so:
    ab=0.59#attenuation in liver
    #rf=10*np.log10(rf2/phantom)
    q=6/z#0.3#6
    '''sanse: transducer-dependent parameter, ratio of imaging depth to radius of transducer'''
    '''sanse: hard-coded parameters are transducer-dependent or mathematical constants'''
    L=15*np.shape(rf)[1]/sz#0.3 #length of window. based on personal theory, 3mm is best. most formulas account for math in cm
    #f=np.linspace(freq-2,freq+2,num=rf.shape[0])
    #S=np.log((10*rf)/(10*(f.reshape((rf.shape[0],1)))**4))#rf-np.log10((10*(f.reshape((rf.shape[0],1)))**4))#np.log((10*rf)/(10*(f.reshape((rf.shape[0],1)))**4))
    #print(f.dtype)
    #print(np.shape(S))
    #[a, b]=np.polyfit(f,S,1)#[a, b]=np.polyfit(np.asarray(f),np.asarray(np.float64(S)),1)#[a, b]=np.polyfit(f,S,1)
    freq=freq#*1000000
    
    sl=a#/freq**2
    ESD = 2*(sl/((11.6*q**2)+52.8))**0.5
    
#     As=15.7*(10**-17)*freq**2
#     B0=ab-As/freq
#     EAC=64*((10**((b)/10))/185*L*q*ESD**6)#64*((10**((b+2*z*B0*freq)/10))/185*L*q*ESD**6)
    
    
    As=0.5
    B0=(ab-As)#/10000
    #EAC=64*((10**((b+2*z*B0*freq)/10))/185*L*q*ESD**6)
    EAC=64*((10**((b+2*z*B0*freq)/10))/(185*L*(q**2)*ESD**6))
    #EAC=64*((10**((b)/10))/(185*L*(q**2)*ESD**6))

    
    return [np.abs(ESD), np.abs(EAC)]#[np.hstack((ESD, EAC))]

#@nb.njit(fastmath=True)
def ESDAC4(phantom,rf,freq,p,ab,sz,a,b):
    '''sanse: EASDAC3 -> uses theoretical att. of liver, EASDAC4 -> uses calculated
        they seem to provide complementary pieces of information for AI'''
    #same as above but uses calculated attenuation
    #p= points, as the liver is 3904 points deep and 15cm, depth=15*p/3904
    z=15*p/sz#3904
    #according to the documents I was given, bsc in phantom =10^-3
    #attenuation coefficient = 0.5 for phantom at 3 mhz 
    #as= attenuation coefficient of saline, approximate as water using: As=15.7*10^-17*f^2
    #bandwith=4 mhz?
    #q is the ratio of depth to aperture radius (or something), evidently 6 is a safe number to use?
    #if we ever use propietary devices, I want that!!
    #if so:
    #ab=0.59#attenuation in liver
   # rf=10*np.log10(rf2/phantom)
    q=6/z#0.3#6
    freq=freq#*1000000
    L=15*np.shape(rf)[1]/sz#0.3 #length of window. based on personal theory, 3mm is best. most formulas account for math in cm
    #f=np.linspace(freq-2,freq+2,num=rf.shape[0])
    #S=np.log((10*rf)/(10*(f.reshape((rf.shape[0],1)))**4))#rf-np.log10((10*(f.reshape((rf.shape[0],1)))**4))#np.log((10*rf)/(10*(f.reshape((rf.shape[0],1)))**4))
    #[a, b]=np.polyfit(f,S,1)#[a, b]=np.polyfit(np.asarray(f),np.asarray(np.float64(S)),1)#np.polyfit(f,S,1)
    sl=a#/freq**2
    ESD = 2*(sl/((11.6*q**2)+52.8))**0.5
    
#     As=15.7*(10**-17)*freq**2
#     B0=ab-As/freq
#     EAC=64*((10**((b)/10))/185*L*q*ESD**6)#64*((10**((b+2*z*B0*freq)/10))/185*L*q*ESD**6)
    
    As=0.5
    B0=(ab-As)#/10000
    #EAC=64*((10**((b+2*z*B0*freq)/10))/185*L*q*ESD**6)
    #EAC=64*((10**((b+2*z*B0*freq)/10))/(185*L*(q**2)*ESD**6))
    #EAC=64*((10**((b)/10))/(185*L*(q**2)*ESD**6))
    EAC=64*((10**((b+2*z*B0*freq)/10))/(185*L*(q**2)*ESD**6))


    return  [np.abs(ESD), np.abs(EAC)]#[np.hstack((ESD, EAC))]


import copy
def treeCreator3Clar81(r,Phantom5, freq,depth,ham,width,freq_num_for_att):
    '''sms: qus metrics except for ps'''
    bsc3=Tree()
    n=0#would be relevant if we ever figured out moving windows
    b=depth#61#122#64 #vertical window length
    l=0#inital depth of window
    h=0# would be relevant if I figured out moving windows
    nat=setforrad3(r) # creating the quantised dat in a pseudo guassian distribution, not the fastest but increadibly important for disease detection
    sz=np.shape(r)[0]#the number of pixels vertically, important for size calculations
    m=0#initializing a iterating variable
    c=width#16#4#8#4 #horozontal window size
    qa=int(sz/depth)
    wlo=int(np.shape(r)[1]/width)
    #the next few parts are bit hard to explain, but:
    #this is in one line converting the 2d RF (and quantized and phantom) data (that is in clarius' case a 2928 x 192)
    #into a 3d array where the first dimension is the frame number,so specifically a 2304x61x4
    #this allows all the math to become vectorizable (practically)
    rf=np.array([(r[(l+i*b)-n:(l+i*b)+b,(w*c):(w*c+c)]) for i in range(0,qa) for w in range(0,wlo) ])#np.array([((r[(l+i*b)-n:(l+i*b)+b,(w*c):(h*c+c)] )for w in range(0,48) )for i in range(0,48)])
    #print(l+i*b)
    l=0
    natrf=np.array([(nat[(l+i*b)-n:(l+i*b)+b,(w*c):(w*c+c)]) for i in range(0,qa) for w in range(0,wlo) ])#np.array([((nat[(l+i*b)-n:(l+i*b)+b,(w*c):(h*c+c)] )for w in range(0,48) )for i in range(0,48)])
    '''sms: quantitative version of rf data, currently not used.'''
    l=0
    phantrf=np.array([(Phantom5[(l+i*b)-n:(l+i*b)+b,(w*c):(w*c+c)]) for i in range(0,qa) for w in range(0,wlo) ])#np.array([((Phantom5[(l+i*b)-n:(l+i*b)+b,(w*c):(h*c+c)] )for w in range(0,48) )for i in range(0,48)])
    l=0
    h=0
    scz=np.shape(rf)[0] #getting the number of frames
    #vectorized fast fourier transform of the RF data
    rfPSRec=np.fft.fft(rf,axis=1)#np.array([(np.fft.fft(rf[m,:,:],axis=0))for m in range(0,scz)])
    #ibid, for phantom
    pPSRec=np.fft.fft(phantrf,axis=1)#np.array([(np.fft.fft(phantrf[m,:,:],axis=0))for m in range(0,scz)])
    #vectorized fast fourier transform of the RF data with a hamming window applied
    rfPSHam=np.fft.fft((rf[:,:,:]*ham),axis=1)
    #ibid for phantom
    pPSHam=np.fft.fft((phantrf[:,:,:]*ham),axis=1)
    '''sms: this step is done twice in a separate function (rfps), due to convenience'''

    #Nakagami parameters claculated simeltaniously for all windows with vectorization techniques
    Nak=np.asarray(NakaGamiParam2(rf,phantrf))
    #longl is an 1d array the size as number of windows. each value contains the starting pixel number for the corresponding window
    #this is necessary for any vectorized calculation involving depth
    #N.B.: i is vertical, w is horozontal
    longl=np.array([(depth*i) for i in range(0,qa) for w in range(0,wlo) ])
    '''sms: know depth of windows within patient for context information'''

    #vectorized calculation of backscatter coefficient, for a full explanation check out the function
    coeff=BSCcalc2(pPSRec,rfPSRec,freq,longl.reshape((-1,1,1)),sz)#np.array([(BSCcalc2(pPSRec[m,:,:],rfPSRec[m,:,:],freq,longl[m],sz))for m in range(0,scz)])
    #this variable is neccesary for any calcualtion requiring polyfit. corresponds to x values theoretically
    f=np.linspace(freq-2,freq+2,num=rfPSRec.shape[1])
    #S is an intermidiate value for calculating coefficients needed for ESD and EAC, fully vectorized for speed
    #for a full explantion look at old code or read the documents
    
    rfPSRec2=rfPSRec[:,0:int(np.floor(np.shape(rfPSRec)[1]/2)),:]
    pPSRec2=pPSRec[:,0:int(np.floor(np.shape(pPSRec)[1]/2)),:]
    #f2=np.linspace(freq*1000000-2*1000000,freq*1000000+2*1000000,num=pPSRec2.shape[1])
    f2=np.linspace(freq-2,freq+2,num=pPSRec2.shape[1])
    S=10*np.log10(rfPSRec2/pPSRec2)-10*np.log10(f2**4).reshape((-1,1))
    mb=int(np.floor(np.shape(rfPSRec2)[1]/8))
    ESD3=np.array([(np.polyfit(f2[mb*2:]**2,S[m,mb*2:,:],1)) for m in range(0,scz)])
    '''sanse: part of calculation of ESD takes place here (S, mb are intermediate variables)'''
    
    
    
    #S=np.log((10*rfPSRec)/(10*(f.reshape((rfPSRec.shape[1],1)))**4))
    #polyfit is not vectorizable. The returned values are needed for calculating ESD and EAC
    #ESD3=np.array([(np.polyfit(f,S[m,:,:],1)) for m in range(0,scz)])
    #fully vectorized calcuation of ESD and EAC for a rec window retuned shape is [2, nframes, 4]
    #first value is ESD, second is EAC. Check code for more info, or read the old docs
    ESD=np.asarray(ESDAC3(pPSRec,rfPSRec,freq,longl.reshape((-1,1)),sz,ESD3[:,0,:],ESD3[:,1,:]))#np.array([(ESDAC(pPSRec[m,:,:],rfPSRec[m,:,:],freq,longl[m],sz))for m in range(0,scz)])
    
    rfpsrec2=rfPSRec[:,0:int(np.floor(np.shape(rfPSRec)[1]/2)),:]
    pPSrec2=pPSRec[:,0:int(np.floor(np.shape(pPSRec)[1]/2)),:]

    mba=int(np.floor(np.shape(rfpsrec2)[1]*(3-2)/4))#0.325))
    mbb=int(np.floor(np.shape(rfpsrec2)[1]*(5-2)/4))
    '''sanse: section of power spectrum analyzed'''

    f2=np.linspace(freq-2,freq+2,num=pPSrec2.shape[1])
    ## Narrow frequency selection
    #freqs_cutoffs, key_ind = np.unique(np.around(f2,1),return_index=True)
    f3=f2[mba:mbb]#[key_ind[np.nonzero(freqs_cutoffs==lfreq)[0][0]]:key_ind[np.nonzero(freqs_cutoffs==hfreq)[0][0]]]

    #intermidiate (fully vecotrized) value for calculating Lizzie parametrs (for rec window)
    '''sanse: Lizzie parameters - equivalent to QUS (SS, MBF) parameters'''
    S=np.abs(np.log((rfpsrec2)/(pPSrec2)))#np.abs(np.log((rfPSRec)/(pPSRec)))#Add 2 back in if you are curios
    ## Narrow S to S2 same as f3
    S2=S[:,mba:mbb,:]#[:,key_ind[np.nonzero(freqs_cutoffs==lfreq)[0][0]]:key_ind[np.nonzero(freqs_cutoffs==hfreq)[0][0]],:]

    #non vectorizable calculation of SS and SI
    SS=np.array([(np.polyfit(f3,S2[m,:,:],1)) for m in range(0,scz)])#LizziParam3(f,S)
    #Calculation of MBF
    #MBF=SS[:,0,:]*(f3[int(0.5*int(np.floor(np.shape(rfPSRec)[1])))])+SS[:,1,:]#(int(0.5*rfPSRec.shape[1]))+SS[:,1,:]
    MBF=SS[:,0,:]*(f3[int(0.5*len(f3))])+SS[:,1,:]
    '''SS: Spectral intercept/Slope, MBF: Mid-band fit'''

    #concatenation of all Lizzie parameters [SS,SI,MBF]
    LizziRec=np.concatenate((SS,MBF.reshape((-1,1,width))), axis=1)#np.array([(LizziParam(pPSRec[m,:,:],rfPSRec[m,:,:],freq,longl[m],sz))for m in range(0,scz)])

    #similar to the earlier used f, different size b/c windows used for atten calc must be smaller
    f4=np.linspace(freq-2,freq+2,num=freq_num_for_att)
    #intermidiate (fully vectorized) calculation for attenuation coefficient


    TGC_1 = np.linspace(15,35,len(range(0,int(np.floor((12/15)*2928)))))
    TGC = np.concatenate((TGC_1,[35]*len(range(int(np.floor((12/15)*2928)),2928))))
    TGC_final = 10**(TGC/20)
    TGC_final_2D = np.transpose([TGC_final]*192)
    '''sanse: TGC is not anymore necessary, was used in older versions (even with phantom)'''    

    depthb=600
    bb=600
    r2=copy.deepcopy(r)#/TGC_final_2D
    gurf=np.zeros((2928,48))
    shelp=np.zeros((2928,48))
    for hj in range(0,17):
        nerg=hj*33
        '''sanse: this is for overlapping windows - only used for attenuation. since optimum window size for attenuation 
        did not match the window size of other parameters or was divisible by total frame size'''
        qb=4#1800+nerg#int(sz/depth)
        #print(qa)
        #print(wlo)
        l=nerg
        #the next few parts are bit hard to explain, but:
        #this is in one line converting the 2d RF (and quantized and phantom) data (that is in clarius' case a 2928 x 192)
        #into a 3d array where the first dimension is the frame number,so specifically a 2304x61x4
        #this allows all the math to become vectorizable (practically)
        rfa=np.array([(r2[(l+i*bb)-n:(l+i*bb)+bb,(w*c):(w*c+c)]) for i in range(0,qb) for w in range(0,wlo) ])#np.array([((r[(l+i*b)-n:(l+i*b)+b,(w*c):(h*c+c)] )for w in range(0,48) )for i in range(0,48)])
        #rf=np.array([(r[(l+i*b)-n:(l+i*b)+b,(w*c):(w*c+c)]) for i in range(0,qa) for w in range(0,wlo) ])#
        #print(l+i*b)
        l=nerg
        longla=np.array([(depthb*i) for i in range(0,qb) for w in range(0,wlo) ])+l
        l=nerg
        phantrfa=np.array([(Phantom5[(l+i*bb)-n:(l+i*bb)+bb,(w*c):(w*c+c)]) for i in range(0,qb) for w in range(0,wlo) ])#np.array([((Phantom5[(l+i*b)-n:(l+i*b)+b,(w*c):(h*c+c)] )for w in range(0,48) )for i in range(0,48)])
        l=0
        h=0
        scza=np.shape(rfa)[0] #getting the number of frames
        f4=np.linspace(freq-2,freq+2,num=100)#25
        mba=int(np.floor(np.shape(f4)[0]*0.25))#0.325))#375
        mbb=int(np.floor(np.shape(f4)[0]*0.5))#0.75))#675
        #intermidiate (fully vectorized) calculation for attenuation coefficient
        tepiid=AttenuationCoeff42(phantrfa,rfa,freq,longla.reshape((-1,1,1)),sz,f4,200)
        atten=np.nanmean(np.abs(np.array([(np.polyfit(f4[mba:mbb],tepiid[m,:,:],1)) for m in range(0,scza)])[:,0,:]),axis=1)
        
        ra=np.reshape(atten,(-1,48))
        helk=np.repeat(ra,600,axis=0)
        hrm=nerg
        
        gurf[hrm:2400+hrm,:]=gurf[hrm:2400+hrm,:]+helk
        '''sanse: each time we iterate and calculate new att. coefficient'''
        shelp[hrm:2400+hrm,:]=shelp[hrm:2400+hrm,:]+1
        '''sanse: keeps track of number of att coefficients added together at each pixel'''
        '''sanse: this is related to overlaps'''
        
    burt=gurf/shelp
    attenp=np.zeros((qa,wlo))
    hef=0
    for i in range(0,qa):
        #attenp[i,:]=np.mean(burt[hef:hef+61,:],axis=0)
        #hef=hef+61
        attenp[i,:]=np.mean(burt[hef:hef+b,:],axis=0)
        hef=hef+b
    #srp3.append(attenp)
    atten=np.reshape(attenp,(-1,))
#     f4=np.reshape(f4,(25,1))
#     f4_rep = np.concatenate((f4,f4,f4,f4),axis=1)
#     atten=np.nanmean(np.abs(np.array([tepiid[m,:,:]/f4_rep for m in range(0,scz)])),axis=1).reshape(-1,1,4)
#     atten=np.nanmean(atten,axis=2)
    #fully vectorized calculation of backscatter coefficient for hamming window with calculated attenuation coefficient
    
    atten2=copy.deepcopy(np.reshape(atten,(qa,wlo)))
    atten3=np.array([(np.nanmean(atten2[:m+1,:],axis=0)) for m in range(0,qa)])
    atten3=np.reshape(atten3,np.shape(atten))
    
    
    coeffHam=BSCcalc3(pPSHam,rfPSHam,freq,longl.reshape((-1,1,1)),atten3.reshape((-1,1,1)),sz)
    #coeffHam=BSCcalc3(pPSHam,rfPSHam,freq,longl.reshape((-1,1,1)),atten.reshape((-1,1,1)),sz)#np.array([(BSCcalc3(pPSHam[m,:,:],rfPSHam[m,:,:],freq,longl[m],atten[m],sz))for m in range(0,scz)])
    
    
    rfPSRec2=rfPSHam[:,0:int(np.floor(np.shape(rfPSHam)[1]/2)),:]
    pPSRec2=pPSHam[:,0:int(np.floor(np.shape(pPSHam)[1]/2)),:]
    #f2=np.linspace(freq*1000000-2*1000000,freq*1000000+2*1000000,num=pPSRec2.shape[1])
    f2=np.linspace(freq-2,freq+2,num=pPSRec2.shape[1])

    S=10*np.log10(rfPSRec2/pPSRec2)-10*np.log10(f2**4).reshape((-1,1))
    mb=int(np.floor(np.shape(rfPSRec2)[1]/8))

    ESD3=np.array([(np.polyfit(f2[mb*2:]**2,S[m,mb*2:,:],1)) for m in range(0,scz)])
    ESDHam=np.asarray(ESDAC4(pPSRec2,rfPSRec2,freq,longl.reshape((-1,1)),atten3.reshape((-1,1)),sz,ESD3[:,0,:],ESD3[:,1,:]))

    rfpsrec2=rfPSHam[:,0:int(np.floor(np.shape(rfPSHam)[1]/2)),:]
    pPSrec2=pPSHam[:,0:int(np.floor(np.shape(pPSHam)[1]/2)),:]

    mba=int(np.floor(np.shape(rfpsrec2)[1]*(3-2)/4))#0.325))
    mbb=int(np.floor(np.shape(rfpsrec2)[1]*(5-2)/4))
    
    f2=np.linspace(freq-2,freq+2,num=pPSrec2.shape[1])
    ## Narrow frequency selection
    #freqs_cutoffs, key_ind = np.unique(np.around(f2,1),return_index=True)
    f3=f2[mba:mbb]#f2[key_ind[np.nonzero(freqs_cutoffs==lfreq)[0][0]]:key_ind[np.nonzero(freqs_cutoffs==hfreq)[0][0]]]
    
    #intermidiate (fully vecotrized) value for calculating Lizzie parametrs (for rec window)
    S=np.abs(np.log((rfpsrec2)/(pPSrec2)))#np.abs(np.log((rfPSRec)/(pPSRec)))#Add 2 back in if you are curios
    ## Narrow S to S2 same as f3
    S2=S[:,mba:mbb,:]#[:,key_ind[np.nonzero(freqs_cutoffs==lfreq)[0][0]]:key_ind[np.nonzero(freqs_cutoffs==hfreq)[0][0]],:]

    #non vectorizable calculation of SS and SI
    SS=np.array([(np.polyfit(f3,S2[m,:,:],1)) for m in range(0,scz)])#LizziParam3(f,S)
    #Calculation of MBF
    #MBF=SS[:,0,:]*(f3[int(0.5*int(np.floor(np.shape(rfPSRec)[1])))])+SS[:,1,:]#(int(0.5*rfPSRec.shape[1]))+SS[:,1,:]
    MBF=SS[:,0,:]*(f3[int(0.5*len(f3))])+SS[:,1,:]

    #concatenation of all Lizzie parameters [SS,SI,MBF]
    LizziHam=np.concatenate((SS,MBF.reshape((-1,1,width))), axis=1)#np.array([(LizziParam(pPSRec[m,:,:],rfPSRec[m,:,:],freq,longl[m],sz))for m in range(0,scz)])
    
    m=0#initilize iterating variable
    for i in range(0,scz):#32):#61):
        bsc3[m].rf=natrf[m,:,:]#nat[l-n:l+b,h:h+c]
        bsc3[m].rfPSRec=rfPSRec[m,:,:]#np.fft.fft(rf[m,:,:],axis=0)#we[l-36:l+36,:]
        bsc3[m].pPSRec=pPSRec[m,:,:]#np.fft.fft(phantrf[m,:,:],axis=0)#,0],axis=0)#wp[l-36:l+36,:]
        bsc3[m].rfPSHam=rfPSHam[m,:,:]#np.fft.fft(rf[m,:,:]*ham,axis=0)#we[l-36:l+36,:]
        bsc3[m].pPSHam=pPSHam[m,:,:]#np.fft.fft(phantrf[m,:,:]*ham,axis=0)#,0]*ham,axis=0)#wp[l-36:l+36,:]
        bsc3[m].Nak=Nak[:,m]#NakaGamiParam(rf[m,:,:],phantrf[m,:,:])#,0])#bsc3[m].rfPSHam,bsc3[m].pPSHam)#r[l-n:l+b,h:h+4,0],Phantom5[l-n:l+b,h:h+4,0])
            #print(l)
        bsc3[m].coeff=coeff[m,:,:]#BSCcalc2(bsc3[m].pPSRec,bsc3[m].rfPSRec,freq,l,sz)
        bsc3[m].ESD=ESD[:,m,:]#ESDAC(bsc3[m].pPSRec,bsc3[m].rfPSRec,freq,l,sz)
        bsc3[m].LizziRec=LizziRec[m,:,:]# The one to run on the Whole data-set #LizziParam(bsc3[m].pPSRec,bsc3[m].rfPSRec,freq,l,sz)

        bsc3[m].atten=atten[m]#AttenuationCoeff(phantrf[m,:,:],rf[m,:,:],freq,l,sz)#,0],rf[m,:,:],freq,l,sz)
        bsc3[m].coeffHam=coeffHam[m,:,:]#BSCcalc3(bsc3[m].pPSHam,bsc3[m].rfPSHam,freq,l,bsc3[m].atten,sz)
        bsc3[m].ESDHam=ESDHam[:,m,:]#ESDAC2(bsc3[m].pPSHam,bsc3[m].rfPSHam,freq,l,bsc3[m].atten,sz)
        bsc3[m].LizziHam=LizziHam[m,:,:]#The one to run on Whole data-set #LizziParam(bsc3[m].pPSHam,bsc3[m].rfPSHam,freq,l,sz)

        m=m+1

    ar3=np.array([(np.abs(np.hstack((np.nanmean(bsc3[i].coeff),np.nanmean(bsc3[i].ESD[0]),np.nanmean(bsc3[i].ESD[1]),np.nanmean(bsc3[i].LizziRec[0]),
    np.nanmean(bsc3[i].LizziRec[1]),np.nanmean(bsc3[i].LizziRec[2]),bsc3[i].atten,
    np.nanmean(bsc3[i].coeffHam),np.nanmean(bsc3[i].ESDHam[0]),np.nanmean(bsc3[i].ESDHam[1]),np.nanmean(bsc3[i].LizziHam[0]),
    np.nanmean(bsc3[i].LizziHam[1]),np.nanmean(bsc3[i].LizziHam[2]),bsc3[i].Nak,np.nanstd(bsc3[i].ESD[0]),np.nanstd(bsc3[i].ESDHam[0]))))) for i in range(0,m)])

    return (ar3,natrf)#(ar3, armap, bsc3) #(ar1, ar2, ar3)


def NPSCreator3Clar81(r,Phantom5, freq,depth,hanning,width):
    #Initialize the tree that will hold the intermidiate values
    #I like them
    bsc4=Tree()
    bsc3=Tree()
    n=0#would be relevant if we ever figured out moving windows
    b=depth#61#122#64 #vertical window length
    l=0#inital depth of window
    h=0# would be relevant if I figured out moving windows
    #nat=setforrad3(r) # creating the quantised dat in a pseudo guassian distribution, not the fastest but increadibly important for disease detection
    sz=np.shape(r)[0]#the number of pixels vertically, important for size calculations
    m=0#initializing a iterating variable
    c=width#16#4#16#4#8#4 #horozontal window size
    qa=int(sz/depth)
    wlo=int(np.shape(r)[1]/width)
    #the next few parts are bit hard to explain, but:
    #this is in one line converting the 2d RF (and quantized and phantom) data (that is in clarius' case a 2928 x 192)
    #into a 3d array where the first dimension is the frame number,so specifically a 2304x61x4
    #this allows all the math to become vectorizable (practically)
    
    #tempPhant=np.mean(Phantom5[:,90:100],axis=1)
    
    tempPhant=np.mean(Phantom5[:,90:100],axis=1)
    #print(np.shape(tempPhant))
    tempPhant=np.reshape(tempPhant,(-1,1))
    Phantom6=np.repeat(tempPhant,192, axis=1)
    
    rf=np.array([(r[(l+i*b)-n:(l+i*b)+b,(w*c):(w*c+c)]) for i in range(0,qa) for w in range(0,wlo) ])#np.array([((r[(l+i*b)-n:(l+i*b)+b,(w*c):(h*c+c)] )for w in range(0,48) )for i in range(0,48)])
    l=0
    '''sms: each frame makes 3D, no. windows, height, weight'''

    #natrf=np.array([(nat[(l+i*b)-n:(l+i*b)+b,(w*c):(w*c+c)]) for i in range(0,48) for w in range(0,48) ])#np.array([((nat[(l+i*b)-n:(l+i*b)+b,(w*c):(h*c+c)] )for w in range(0,48) )for i in range(0,48)])
    l=0
    phantrf=np.array([(Phantom5[(l+i*b)-n:(l+i*b)+b,(w*c):(w*c+c)]) for i in range(0,qa) for w in range(0,wlo) ])#np.array([((Phantom5[(l+i*b)-n:(l+i*b)+b,(w*c):(h*c+c)] )for w in range(0,48) )for i in range(0,48)])
    phantrf2=np.array([(Phantom6[(l+i*b)-n:(l+i*b)+b,(w*c):(w*c+c)]) for i in range(0,qa) for w in range(0,wlo) ])
    '''sms: there were different versions of the phantom, 
    Phantom 5: taking all lines as measured in to account with potential edge effects due to limited phantom size
    Phantom 6: repeating central line
    At the moment phantoms are not used for ps.'''

    #print(np.shape(rf))
    l=0
    h=0
    scz=np.shape(rf)[0] #getting the number of frames
    
    #vectorized fast fourier transform of the RF data
    #rfPSRec=np.fft.fft(rf,axis=1)#np.array([(np.fft.fft(rf[m,:,:],axis=0))for m in range(0,scz)])
    #ibid, for phantom
    #pPSRec=np.fft.fft(phantrf,axis=1)#np.array([(np.fft.fft(phantrf[m,:,:],axis=0))for m in range(0,scz)])
    
    #vectorized fast fourier transform of the RF data with a hamming window applied
    rfPSHan=np.fft.fft((rf[:,:,:]*hanning),axis=1)
    
    rfPSHan=rfPSHan[:,0:int(np.floor(np.shape(rfPSHan)[1]/2)),:]
    #print(np.shape(rfPSHan))
    #ibid for phantom
    pPSHan=np.fft.fft((phantrf2[:,:,:]*hanning),axis=1)
    pPSHan=pPSHan[:,0:int(np.floor(np.shape(pPSHan)[1]/2)),:]
    
    rfPSHan=rfPSHan**2
    rfPSHan=np.abs(rfPSHan)
    rfPSHan=np.mean(rfPSHan,axis=2)
    rfPSHan=10*np.log10(rfPSHan)
    '''sms: break into depth windows without overlap, window size: match qus windows (experimental),
       take for lines (122 x 4) -> Average in width -> (122 x 1)
       phase information is not considered   
    '''


    pPSHan=pPSHan**2
    pPSHan=np.abs(pPSHan)
    pPSHan=np.mean(pPSHan,axis=2)
    #print(np.shape(rfPSHan))
    pPSHan=10*np.log10(pPSHan)
    
    
    rfnps=rfPSHan-pPSHan
    
    rfps=np.reshape(rfPSHan,(-1, wlo))
    rfnps=np.reshape(rfnps,(-1,wlo))
   
    return (rfps,rfnps)#  bsc4)#(ar3, armap, bsc3) #(ar1, ar2, ar3)


def setforrad3(trial):
    ar=np.zeros(np.shape(trial))#initialize a variable for holding new values
    
    #Read up on quantization
    #non linear homebrew method
    (a,b)=(trial<=-3000).nonzero()
    #at=np.floor((trial-np.min(trial))/(-3000-np.min(trial))*25)
    ar[a,b]=np.floor(((trial[a,b]-np.min(trial))/(-3000-np.min(trial)))*25)#at[a,b]

    #at=(np.floor(((trial-(-3000))/(-1000-(-3000)))*25)+25)

    (a,b)=((-3000<trial) & (trial<=-1000)).nonzero()
    ar[a,b]=(np.floor(((trial[a,b]-(-3000))/(-1000-(-3000)))*25)+25)#at[a,b]


    #at=(np.floor(((trial-(-1000))/(-500-(-1000)))*25)+50)
    (a,b)=((-1000<trial) & (trial<=-500)).nonzero()
    ar[a,b]=(np.floor(((trial[a,b]-(-1000))/(-500-(-1000)))*25)+50)#at[a,b]


    #at=(np.floor(((trial-(-500))/(500-(-500)))*100)+75)
    (a,b)=((-500<trial) & (trial<=500)).nonzero()

    ar[a,b]=(np.floor(((trial[a,b]-(-500))/(500-(-500)))*100)+75)#at[a,b]


    #at=(np.floor(((trial-(500))/(1000-(500)))*25)+175)
    (a,b)=((500<trial) & (trial<=1000)).nonzero()
    ar[a,b]=(np.floor(((trial[a,b]-(500))/(1000-(500)))*25)+175)#at[a,b]



    #at=(np.floor(((trial-(1000))/(3000-(1000)))*25)+200)
    (a,b)=((1000<trial) & (trial<=3000)).nonzero()
    ar[a,b]=(np.floor(((trial[a,b]-(1000))/(3000-(1000)))*25)+200)#at[a,b]

    #at=(np.floor(((trial-(3000))/(np.max(trial)-(3000)))*25)+225)
    (a,b)=(3000<trial).nonzero()
    ar[a,b]=(np.floor(((trial[a,b]-(3000))/(np.max(trial)-(3000)))*25)+225)#at[a,b]
    
    #r=ar[:,:,0]
    #ab=np.ones(np.shape(r))
    

    return (ar)

#################################################################################################################################

# MODIFIED TO WORK WITH GOOGLE CLOUD BUCKETS

def info_Phantom(ID, file):
    
    path = ID + '/extracted/'
    
    Files={}
    Files["directory"] = path
    Files["xmlName"] = "probes_usx.xml"
    #Files["ymlName"] = path + file + "_rf.yml"
    Files["ymlName"] = file + "_rf.yml"
    Files["name"] = file + "_rf.raw"
    imgInfo = read_CLariusinfo_Phantom(Files["name"], Files["xmlName"], Files["ymlName"], Files["directory"])
    
    return imgInfo


# MODIFIED TO WORK WITH GOOGLE CLOUD BUCKETS
def read_CLariusinfo_Phantom(filename=None, xmlFilename=None,ymlFileName=None,  filepath=None):

    import numpy as np
    #from readprobe_CLarius import readprobe_CLarius
    try:
        import yaml
    except:
        print ('You do not have the YAML module installed.\n'+'Run: pip install pyaml to fix this' )
        quit()


    
    # Some Initilization
    studyID = filename[1:(len(filename) - 4)]
    studyEXT = filename[(len(filename) - 2):]

    rfFilePath=filepath+filename
    
    #print(rfFilePath)
    
    # get from google
    while True:
        try:
            temp = storage.blob.Blob(rfFilePath,bucket_pull)
            content = temp.download_as_string()
            break
        except:
            print('\t\tfile load error')
            pass
        
    # write to temp file, then use temp file
    tempFile1 = 'temp_files/tempFile1_Phantom'
    fpoint = open(tempFile1, 'wb')
    fpoint.write(content)
    fpoint.close()

    # Open RF file for reading
    while True:
        try:
            with open(tempFile1, mode='r') as fid : 
                # load the header information into a structure and save under a separate file
                hinfo = np.fromfile(fid , dtype='uint32', count=5 )
            break
        except:
            print('\t\terror opening ' , tempFile1)
            pass
    
    
    
    # delete temp file
    try:
        os.remove(tempFile1)
    except OSError:
        pass
    

    header = {'id': 0, 'frames': 0, 'lines': 0, 'samples': 0, 'sampleSize': 0}
    header["id"] = hinfo[0]
    header["nframes"] = hinfo[1] #frames
    header["w"] = hinfo[2] #lines
    header["h"] = hinfo[3] #samples
    header["ss"] = hinfo[4] #sampleSize

    #from yml file
    # transFreq=yml_data["transmit frequency"]
    # header["txf"] =transFreq[1:(len(transFreq)-3)] # transmit freq - also called center freq  
    header["txf"] =4 

    # sampling_rate=yml_data["sampling rate"]
    # header["sf"] = sampling_rate[1:(len(sampling_rate)-3)]  #sampling freq - also called receive freq = sampling rate
    header["sf"]=20000000
    
    header["dr"] = 23 # Fixed from Usx Probe.xml file
    header["ld"] = 192 # lineDensity => num of lines is 192... standard. 

    info={}

    # For USX - must also read probe file for probe parameters probeStruct and the('probes.xml', header.probe);
    probeStruct = readprobe_CLarius(xmlFilename, 21)
    info["probeStruct"] = probeStruct
    # assignin('base','header', header)

    # Add final parameters to info struct
    info["studyMode"] ="RF"
    info["file"] = filename
    info["filepath"] = filepath
    info["probe"] = "clarius"
    info["system"] = "Clarius"
    info["studyID"] = studyID
    info["samples"] = header["h"]
    info["lines"] = header["w"]#probeStruct.numElements; % or is it > header.w; Oversampled line density?
    info["depthOffset"] = probeStruct["transmitoffset"]# unknown for USX
    info["depth"] = header["ss"] * 10 ** 1 #1275/8; % in mm; from SonixDataTool.m:603 - is it header.dr?
    info["width"] = header["dr"] * 10 ** 1 #1827/8; %info["probeStruct.pitch*1e-3*info["probeStruct.numElements; % in mm; pitch is distance between elements center to element center in micrometers
    info["rxFrequency"] = header["sf"]
    info["samplingFrequency"] = header["sf"]
    info["txFrequency"] = header["txf"]
    info["centerFrequency"] = header["txf"] #should be same as transmit freq - it's the freq. of transducer
    info["targetFOV"] = 0
    info["numFocalZones"] = 1#Hardcoded for now - should be readable
    info["numFrames"] = header["nframes"]
    info["frameSize"] = info["depth"] * info["width"]
    info["depthAxis"] = info["depth"]
    info["widthhAxis"] = info["width"]
    info["lineDensity"] = header["ld"]
    info["height"] = info["depth"]#This could be different if there is a target FOV
    info["pitch"] = probeStruct["pitch"]
    info["dynRange"] = 0# Not sure if available
    info["yOffset"] = 0
    info["vOffset"] = 0
    info["lowBandFreq"] = info["txFrequency"] - 0.5 * probeStruct["frequency"]["bandwidth"]
    info["upBandFreq"] = info["txFrequency"] + 0.5 * probeStruct["frequency"]["bandwidth"]
    info["gain"] = 0
    info["rxGain"] = 0
    info["userGain"] = 0
    info["txPower"] = 0
    info["power"] = 0
    info["PRF"] = 0

    # One of these is the preSC, the other is postSC resolutions
    info["yRes"] = ((info["samples"] / info["rxFrequency"] * 1540 / 2) / info["samples"]) * 10 ** 3#>> real resolution based on curvature
    info["yResRF"] = info["depth"] / info["samples"]#>> fake resolution - simulating linear probe
    info["xRes"] = (info["probeStruct"]["pitch"] * 1e-6 * info["probeStruct"]["numElements"] / info["lineDensity"]) * 10 ** 3 #>> real resolution based on curvature
    info["xResRF"] = info["width"] / info["lines"]#>> fake resolution - simulating linear probe


    # Quad 2 or accounting for change in line density 
    info["quad2X"]= 1

    # Ultrasonix specific - for scan conversion - from: sdk607/MATLAB/SonixDataTools/SonixDataTools.m:719
    info["Apitch"] = (info["samples"] / info["rxFrequency"]* 1540 / 2) / info["samples"]# Axial pitch - axial pitch - in metres as expected by scanconvert.m
    info["Lpitch"] = info["probeStruct"]["pitch"] * 1e-6 * info["probeStruct"]["numElements"] / info["lineDensity"]# Lateral pitch - lateral pitch - in meters
    info["Radius"] = info["probeStruct"]["radius"] * 1e-6
    info["PixelsPerMM"] = 8# Number used to interpolate number of pixels to be placed in a mm in image
    info["lateralRes"] = 1 / info["PixelsPerMM"]# Resolution of postSC
    info["axialRes"] = 1 / info["PixelsPerMM"]# Resolution of postSC

    #print ("Clarius Info : %s " % info)

    return info


def read_Clariusimg_Phantom(Info=None, frame=None):
    import os
    import sys

    #sys.path.append(os.path.dirname('./solution/clarius_read/'))
    #sys.path.append(os.path.dirname('./solution/convert/'))

    #from PRead_Clarius import PRead_Clarius
    #from PRead_Clarius2 import PRead_Clarius2
    #from rf2bmode import rf2bmode
    #from scanconvert_mapped import scanconvert_mapped
    #from scanconvert import scanconvert


    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Read ultrasonix rf data from tablet and touch. Needs meta data stored as struct.
    # The ModeIM struct contains a map of x and y coordiantes to retrace a
    # point in a scan conerted image back to the original non-scanconverted RF
    # data.
    #Input:
    # Info - meta data with parameters for image, analysis and display
    # frame - frame number to read
    #Output:
    # Bmode - Scan converted bmode image for display. 
    # ModeIM - Contains: .orig (original RF data for image), .data (scan converted RF data), .xmap (x coordinates of point on original data), .ymap (ycoordinate of point of original data)
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # Get RF data file path using the metadata in Info
    usx_rf_filepath = Info["filepath"]+Info["file"]

    # Read the image data
    ModeIM=PRead_Clarius_Phantom(usx_rf_filepath, '6.0.3' )

    frame=frame-1
    
    # Make ModeIM just one frame - the chosen frame
    ModeIM = ModeIM[:,:, frame]

    # frame=frame-1
    # Create a straightforward Bmode without scan conversion for frame number frame
    Bmode = rf2bmode(ModeIM)# pre-scan converted b-mode
    #print(np.shape(Bmode))
    # Get the map of coordinates xmap and ymap to be able to plot a point in
    # scanconvert back to original
    #scModeIM =scanconvert(ModeIM, Info)# scanconvert_mapped(ModeIM, Info)

    # Simple scan convert of the image to be displayed. 
    scBmode = scanconvert(Bmode, Info)

    # Ouput
    Data={}
    #Data["scRF"] = scModeIM
    Data["scBmode"] = scBmode
    Data["RF"] = ModeIM
    #Data["Bmode"] = Bmode

    return Data


def PRead_Clarius_Phantom(filename,version='6.0.3'): 
    
    # Some Initilization
    studyID = filename[1:(len(filename) - 4)]
    studyEXT = filename[(len(filename) - 2):]
    
    # get from google
    while True:
        try:
            temp = storage.blob.Blob(filename,bucket_pull)
            content = temp.download_as_string()
            break
        except:
            print('\t\tfile load error')
            pass
        
    # write to temp file, then use temp file
    tempFile4 = 'temp_files/tempFile4_Phantom'
    fpoint = open(tempFile4, 'wb')
    fpoint.write(content)
    fpoint.close()
    
    while True:
        try:
            fid = open(tempFile4,  mode='rb')
            break
        except:
            print('\t\terror opening ' , tempFile4)
            pass
    
    

# read the header info
    hinfo = np.fromfile(fid , dtype='uint32', count=5 )
    header = {'id': 0, 'nframes': 0, 'w': 0, 'h': 0, 'ss': 0}
    header["id"] = hinfo[0]
    header["nframes"] = hinfo[1] #frames
    header["w"] = hinfo[2] #lines
    header["h"] = hinfo[3] #samples
    header["ss"] = hinfo[4] #sampleSize


# % ADDED BY AHMED EL KAFFAS - 22/09/2018
    frames = header["nframes"]

    id = header["id"]
    if(id==0):  #iq
        ts = np.zeros(shape=(1,frames))
        data = np.zeros(shape=(frames, header["h"]*2, header["w"]))
    #  read ENV data
        for f in range (frames): 
        # read time stamp
            ts[f] = np.fromfile(fid , dtype='uint64', count=1 )
        # read one line
            oneline = np.fromfile(fid, dtype='uint16').reshape((header["h"]*2, header["w"])).T
            data[f,:,:] = oneline
#######################################################################################################
    elif(id==1): #env
        ts = np.zeros(shape=(1,frames))
        data = np.zeros(shape=(frames, header["h"], header["w"])) 
    #  read ENV data
        for f in range (frames): 
        # read time stamp
            ts[f] = np.fromfile(fid , dtype='uint64', count=1 )
        # read one line
            oneline = np.fromfile(fid, dtype='uint8').reshape((header["h"], header["w"])).T
            data[f,:,:] = oneline
#######################################################################################################
    elif(id==2): #RF
        ts = np.zeros(shape=(1,frames))
        data = np.zeros(shape=(header["h"], header["w"],frames))
    #  read RF data
        for f in range (frames):    
            v = np.fromfile(fid, count=header["h"]*header["w"] , dtype='int16' )
            data[:,:,f] =np.flip(v.reshape(header["h"], header["w"],order ='F').astype(np.int16), axis=1)
#######################################################################################################
    elif(id==3):
        ts = np.zeros(shape=(1,frames))
        data = np.zeros(shape=(frames, header["h"]*2, header["w"])) 
    #  read  data
        for f in range (frames): 
            # read time stamp
            ts[f] = np.fromfile(fid , dtype='int64', count=1 )
            # read one line
            oneline = np.fromfile(fid, dtype='int16').reshape((header["h"]*2, header["w"])).T
            data[f,:,:] = oneline
    
    # delete temp file
    try:
        os.remove(tempFile4)
    except OSError:
        pass

    return data

#################################################################################################################################


#################################################################################################################################

# calls the relevant functions to generate nps, rf, bmode, and qus data, then saves them to a google cloud bucket
def save_the_whales_videos_2(ID_List,depth,width,ham,hanning,freq_num_for_att):
    '''sms: freq_num_for_att not used'''
    #print('1')
    for ID in tqdm(ID_List):
        
        try:
        
            print('\n',ID, '\t--------------------------------------------------------------------------------------')
            # get all file names for the given CASE ID
            file_list = files_in_ID(ID)
            
            print('\t\t\tnumber of videos: ', len(file_list))

            # use all cores
            num_cores = multiprocessing.cpu_count()
            # read compute is our other function that calls a ton of the other confusing functions
            #results = Parallel(n_jobs=num_cores-1,mmap_mode='r+')(delayed(read_compute)(ID,file) for file in file_list)

            ##########################################################################################################################

            #'loky', 'threading', 'multiprocessing'

            # loky is garbage and doesnt work with this code
            #with parallel_backend('loky', inner_max_num_threads=2):
            #    results = Parallel(n_jobs=2,mmap_mode='r+')(delayed(read_compute)(ID,file) for file in file_list)

            # threading WORKS
            #with parallel_backend('threading'):
            #    results = Parallel(n_jobs=2,mmap_mode='r+')(delayed(read_compute)(ID,file) for file in file_list)

            # multiprocessing
            with parallel_backend('multiprocessing'):
                results = Parallel(n_jobs=num_cores-1,mmap_mode='r+')(delayed(read_compute)(ID,file,depth,width,ham,hanning,freq_num_for_att) for file in file_list)
            '''sms: read_compute is where qus takes place'''
            ##########################################################################################################################

            print('\tTIME TO UPLOAD')

            with parallel_backend('multiprocessing'):
                results2 = Parallel(n_jobs=num_cores,mmap_mode='r+')(delayed(save_video_data)( ID , file_list[qq] , results[qq],depth, width ) for qq in range(0, len(results)) )

            print('\t\t\tupload complete')
            try:
                del results
            except:
                pass
        
        except:
            try:
                print('\n',ID, '\t--------------------------------------------------------------------------------------')
                # get all file names for the given CASE ID
                file_list = files_in_ID(ID)

                print('\t\t\tnumber of videos: ', len(file_list))

                # use all cores
                num_cores = multiprocessing.cpu_count()
                # read compute is our other function that calls a ton of the other confusing functions
                #results = Parallel(n_jobs=num_cores-1,mmap_mode='r+')(delayed(read_compute)(ID,file) for file in file_list)

                ##########################################################################################################################

                #'loky', 'threading', 'multiprocessing'

                # loky is garbage and doesnt work with this code
                #with parallel_backend('loky', inner_max_num_threads=2):
                #    results = Parallel(n_jobs=2,mmap_mode='r+')(delayed(read_compute)(ID,file) for file in file_list)

                # threading WORKS
                #with parallel_backend('threading'):
                #    results = Parallel(n_jobs=2,mmap_mode='r+')(delayed(read_compute)(ID,file) for file in file_list)

                # multiprocessing
                with parallel_backend('multiprocessing'):
                    results = Parallel(n_jobs=num_cores-1,mmap_mode='r+')(delayed(read_compute)(ID,file,depth,width,ham,hanning,freq_num_for_att) for file in file_list)

                ##########################################################################################################################

                print('\tTIME TO UPLOAD')

                with parallel_backend('multiprocessing'):
                    results2 = Parallel(n_jobs=num_cores,mmap_mode='r+')(delayed(save_video_data)( ID , file_list[qq] , results[qq],depth, width ) for qq in range(0, len(results)) )

                print('\t\t\tupload complete')
                try:
                    del results
                except:
                    pass
                
            except:
                print('\nERROR WITH CASE: ' , ID)
        
        
        
    return print("Done")



# A FUNCTION TO save the data for each frame of a video with parallel processing
# should speed the code up a lot
def save_video_data(  ID , filename , df  ,depth, width):
    
    
    try:
        # prepare data
        X = np.asarray(df['nps'].tolist())
        X2 = np.asarray(df['ps'].tolist())
        rf = np.asarray(df['rf'].tolist())
        bmode = np.asarray(df['Bmode'].tolist())
        qus = np.asarray(df['qus'].tolist())
        quant = np.asarray(df['quant'].tolist())

        numF = len(X)
        #91=np.floor(depth/2); 
        depthF=int(np.floor(depth/2))
        wlo=int(192/width)#12=192/16
        nps = np.empty((0,numF,depthF,wlo))
        ps = np.empty((0,numF,depthF,wlo))
        # reshape nps
        for j in range(0,int(2928/depth)):##width):
            #nps
            x = X[:,j*depthF:(j+1)*depthF,:]#.reshape(1,5,12,91)#.reshape(5,12,1,91)
            nps = np.append(nps,[x],axis=0)
            #ps
            x2 = X2[:,j*depthF:(j+1)*depthF,:]#.reshape(1,5,12,91)#.reshape(5,12,1,91)
            ps = np.append(ps,[x2],axis=0)
        nps = np.transpose(nps,axes=(1,0,3,2))
        ps = np.transpose(ps,axes=(1,0,3,2))
        
    except:
        print('\t\t\t\tDATA CORRUPTION: ', filename)
        return 0
    
    
#     # NPS
#     while True:
#         try:
#             np.save( file_io.FileIO( ( 'gs://' + bucket_push_name + '/' + ( 'nps_v/' + ('{}_{}_{}{}'.format(ID,filename,'nps','.npy')) ) ), 'w') , nps )
#             break
#         except:
#             print('\t\tfile UPload error')
#             pass
    
    # PS
    while True:
        try:
            np.save( file_io.FileIO( ( 'gs://' + bucket_push_name + '/' + ( 'ps_v/' + ('{}_{}_{}{}'.format(ID,filename,'ps','.npy')) ) ), 'w') , ps )
            break
        except:
            print('\t\tfile UPload error')
            pass

    # RF
    while True:
        try:
            np.save( file_io.FileIO( ( 'gs://' + bucket_push_name + '/' + ( 'rf_v/' + ('{}_{}_{}{}'.format(ID,filename,'rf','.npy')) ) ), 'w') , rf )
            break
        except:
            print('\t\tfile UPload error')
            pass

    # BMODE
    while True:
        try:
            np.save( file_io.FileIO( ( 'gs://' + bucket_push_name + '/' + ( 'bmode_v/' + ('{}_{}_{}{}'.format(ID,filename,'bmode','.npy')) ) ), 'w') , bmode )
            break
        except:
            print('\t\tfile UPload error')
            pass

    #QUS
    while True:
        try:
            np.save( file_io.FileIO( ( 'gs://' + bucket_push_name + '/' + ( 'qus_v/' + ('{}_{}_{}{}'.format(ID,filename,'qus','.npy')) ) ), 'w') , qus )
            break
        except:
            print('\t\tfile UPload error')
            pass
    
#     #QUANT
#     while True:
#         try:
#             np.save( file_io.FileIO( ( 'gs://' + bucket_push_name + '/' + ( 'quant_v/' + ('{}_{}_{}{}'.format(ID,filename,'quant','.npy')) ) ), 'w') , quant )
#             break
#         except:
#             print('\t\tfile UPload error')
#             pass
    
    return 1

#################################################################################################################

def get_info_frame(data_type):
    
    length = len(data_type) + 3
    
    # get all patient IDS - use the rf VIDEO folder for simplicity
    # get the blob iterator objects
    blob = bucket_pull.list_blobs()
    # get relevant file names from blob objects
    fileList1 = [file.name for file in blob if ( file.name.endswith((data_type+'.npy')) and file.name.startswith((data_type+'_v/')) ) ]
    fileList1 = np.asarray(fileList1)
    size = np.size(fileList1)
    
    #ID_list1 = np.copy(fileList1)
    #for ii in range(0, size):
    #    ID_list1[ii] = fileList1[ii][length:length+16]
    # avoid duplicates
    #ID_list1 = np.unique(ID_list1)
    
    # get all video names
    size = np.shape(fileList1)[0]
    fileList = np.copy(fileList1)
    caseID = np.copy(fileList)
    videID = np.copy(fileList)
    bothID = np.copy(fileList)
    frameID = np.copy(fileList)

    for ii in range(0, size):
        fileList[ii] = fileList1[ii][length:]
        split = fileList[ii].split('_')
        caseID[ii] = split[0]
        videID[ii] = split[1]
        bothID[ii] = split[0] + '_' + split[1]
#         frameID[ii] = split[0] + '_' + split[1] + '_' + split[2][:len(data_type)]

    caseID = np.unique(caseID)
    videID = np.unique(videID)
    bothID2 = np.unique(bothID)
#     frameID = np.unique(frameID)
    
    #return caseID, videID, bothID2 , frameID
    
    
    # get videos and frame numbers for each case
    info_list = []
    for ii in range(0, np.shape(caseID)[0]):
        # set up info list
        info_list.append( [] )
        # get ID
        info_list[ii].append(caseID[ii])
        # get video list
        vidlist = [ jj for jj in bothID2 if caseID[ii] in jj ]
        vidlist =  [ jj[17:] for jj in vidlist ]
        info_list[ii].append(vidlist)
        # get number of frames a list for the vids
        framelist = []
        for kk in range(0, np.shape(vidlist)[0]):
            numframes = np.size( [ll for ll in bothID if vidlist[kk] in ll] )
            framelist.append(numframes)
        info_list[ii].append(framelist)
        
#     # get list of frames
#     frameID = []
#     for ii in range(0,len(info_list)):
#         tempID = info_list[ii][0]
#         for jj in range(0,len(info_list[ii][1])):
#             tempVid = info_list[ii][1][jj]
#             for kk in range(0,info_list[ii][2][jj]):
#                 frameID.append( tempID + '_' + tempVid + '_' + str(kk) )
    
    return caseID, videID, bothID2 , frameID , info_list

# pass case ID, video name, number of frames
def load_video_frames( ID , vid , numf , data_type ):
    filenames = []
    for ii in range(0,numf):
        filenames.append( data_type + '/' + ID + '_' + vid + '_' + str(ii) + '_' + data_type + '.npy' )
    
    video = np.zeros((numf,2928,192))
    
    for ii in range(0,numf):
        temp = storage.blob.Blob(filenames[ii],bucket_pull)
        content = temp.download_as_string()
        
        video[ii,:,:] = np.load(BytesIO(content))
    
    return video

# load video directly from _v/ folder
def load_video( ID , vid , data_type):
    
    filename = data_type + '_v/' + ID + '_' + vid + '_' + data_type + '.npy'
    
    temp = storage.blob.Blob(filename,bucket_pull)
    content = temp.download_as_string()
    video = np.load(BytesIO(content))
    
    return video

# input variables are described below
# video : a single entry of RF_list (from patient_loader) - RF_list is the RF (signal) data for all slices of all frames of all videos
# threshold : the LOWER LIMIT of what is considered a normal difference between consecutive signal point, in non-shadowing data. A greater value means more leniency in assigning deadspots (potential shadow)
# deadlength : the region size when searching for deadspots within a signal. 350 seems reasonable as each slice gives ~2800 data points
# shift : the shift applied to regions when scanning through a slice (signal) in search of deadspots. A greater shift means less overlap between consecutive regions. Some overlap is needed as a safety net
# deadavg : used on slices, when looking at regions of size deadlength, with values of 0 (dead) or 1 (alive) for each data point. An average along the region of below "deadavg" implies a deadspot
# shadowing_threshold: threshold for % area of frame that is shadowing

def video_shadowing_quicker( video, threshold, deadlength, shift, deadavg, shadowing_threshold ):
    
    signal_threshold = 100
    
    videoshape = np.shape(video)
    numFrames = videoshape[0]
    slices = videoshape[2]
    numPoints = videoshape[1]
    videoDeadFrames = np.zeros(numFrames)
    
    #number of regions used for each signal
    sizeup = numPoints - 1
    numRegions = (int)(1 + (int) ( (sizeup-deadlength)/shift ) )
    
    deadspots = np.zeros((numFrames , slices , numRegions ))
    deadspots_disp = np.zeros(( numFrames , numPoints, slices ))
    
    #first, find difference between each pair of consecutive points
    diff = np.diff( video , axis = 1 )
    
    #organize each difference as a 1 or 0, wrt some threshold difference we would expect for non-shadowing data
    
    thresh = 1.0 * ( np.abs( diff ) > threshold)
    
    startspot = np.array( [ (shift*ii) for ii in range(0, numRegions) ] )
    endspot = np.add( startspot , deadlength )
    endspot[numRegions-1] = sizeup
    
    avgLife = np.zeros( (numFrames , numRegions , slices) )
    
    for kk in range (0, numFrames):
        avgLife[ kk , :, : ] = np.array( [ [ (np.mean( thresh[ kk , startspot[ii] : endspot[ii]  , jj ] )) for jj in range (0, slices) ] for ii in range( 0 , numRegions ) ]  )
    
    deadspot = np.array(  [ [ [ (( avgLife[kk, ii, jj] < deadavg ) and ( np.mean(np.abs(video[kk, startspot[ii]:endspot[ii]+1 , jj])) < signal_threshold )) for ii in range( 0, numRegions) ] for jj in range(0, slices)]  for kk in range( 0, numFrames) ] )
    deadspot = np.multiply( deadspot , 1 )
    # make an array that simplifies the display of the deadspaces -> if the regions is dead, fill it with 1s
    deadspots_disp = np.zeros( (numFrames, numPoints , slices) )
    
    #fill in deadspots_disp
    for kk in range (0, numFrames):
        for jj in range( 0 , slices ):
            for ii in range( 0 , numRegions ):
                if (deadspot[kk, jj , ii] == 1):
                    deadspots_disp[kk, startspot[ii] :endspot[ii] , jj ] = 1
    
    total = slices*numPoints
    
    dead = np.zeros(numFrames)
    shadowing_fraction = np.zeros(numFrames)
    for kk in range (0, numFrames):
        dead[kk] = np.sum(deadspots_disp[kk,:,:])
        shadowing_fraction[kk] = dead[kk]/total
    
    # ceil to 2 decimal places
    shadowing_fraction = 0.01 * (np.ceil( np.multiply(100.0,shadowing_fraction) ))
    frame_check = [ 1.0*(shadowing_fraction[ii] < shadowing_threshold) for ii in range(0, numFrames) ]## what I need
    
    return frame_check #, video_check, chunkFound

#################################################################################################################################

# THE STUFF THAT NEEDS TO RUN
'''Sergio: Start of the code'''

print('STARTING PREP')

#################################################################################################################################

# set up the buckets
storage_client = storage.Client()

# Define buckets to load data from in list
bucket_list=["pd-cases-tar-extracted","clarius-full","new-cases-1-extract","cases-02_02-22_05"]
'''sms: All the data that is currently available'''
cases_list=["liver_data_6.csv"]#["PD_USA_cases_processed.csv"]#["Combined_Cases_Mar15th2021.csv"]#["pd_cases_tar_extracted_full_list.csv","clarius_full_list.csv","newcases1ext_full_list.csv","cases020202205_full_list.csv"]
'''sms: Selection of patients'''

for casebucket_no in range(0,len(bucket_list)):
    bucket_pull_name = bucket_list[casebucket_no]#"cases-02_02-22_05"#"clarius-full"#"pd-cases-tar-extracted"#"new-cases-1-extract"#"clarius-full"#"pd-cases-tar-extracted"
    print(bucket_pull_name)
    bucket_push_name = "liver-data-raw-nov22-2021"#"us_cases_mar262021"#"segmentation-mar15-2021"#"liver-data-48x61-alldata"#"liver-data-test-2"
    print(bucket_push_name)
    Master_bucket=bucket_push_name
    print(Master_bucket)


    depth=122#61#b=61, 122, 183, 61 
    width=4#c=4, 4, 12, 12
    '''sms: number of lines/depth in window'''
    freq_num_for_att=25#is assigned to "num" when computing "f4" for AttenuationCoeff2
    '''sms: not used'''

    ham=sp.hamming(depth).reshape((depth,1))
    hanning=sp.hann(depth).reshape((depth,1))

    remainingIDs_df = pd.read_csv(cases_list[0])
    remainingIDs=remainingIDs_df["caseID"].to_list()
    print(len(remainingIDs))

    chosen_IDs = remainingIDs

    print(len(chosen_IDs))
    print(chosen_IDs)

    bucket_pull = storage_client.get_bucket(bucket_pull_name)
    bucket_push = storage_client.get_bucket(bucket_push_name)

    #################################################################################################################################

    # get XML file from google
    temp1 = storage.blob.Blob('probes_usx.xml',bucket_pull)
    content1 = temp1.download_as_string()
    # write to temp file, then use temp file
    fpoint1 = open('temp_files/xml_file', 'wb')
    fpoint1.write(content1)
    fpoint1.close()

    print('\t-----got probes_usx.xml-----')

    # get all case IDs that have already been completed
    # get the blob iterator objects
    #blob = bucket_pull.list_blobs()
    blob = bucket_push.list_blobs()
    # get relevant file names from blob objects
    # we only care about case IDs that actually have _rf.raw files
    fileList1 = [file.name for file in blob if ( file.name.endswith("_rf.npy") and file.name.startswith("rf_v/") ) ]
    fileList1 = np.asarray(fileList1)
    size = np.size(fileList1)
    ID_list1 = np.copy(fileList1)
    for ii in range(0, size):
        ID_list1[ii] = fileList1[ii][5:18+3]
    # avoid duplicates
    ID_list1 = np.unique(ID_list1)

    # get all patient IDS
    # get the blob iterator objects
    #blob = bucket_pull.list_blobs()
    blob = bucket_pull.list_blobs(prefix='PDG')
    # get relevant file names from blob objects
    # we only care about case IDs that actually have _rf.raw files
    fileList = [file.name for file in blob if file.name.endswith("_rf.raw")]
    ID_list = np.asarray(fileList)
    size = np.size(ID_list)
    for ii in range(0, size):
        ID_list[ii] = ID_list[ii][:16]
    # avoid duplicates
    ID_list = np.unique(ID_list)
    # trim the _rf.raw
    ID_list = list(ID_list)


    # get phantom stuff
    ID_Phantom = 'Phantom'
    # there is only one file name for the phantom
    file_Phantom = '2019-08-22T00-41-10+0000'
    imgInfo_Phantom = info_Phantom(ID_Phantom, file_Phantom)
    D2 = read_Clariusimg_Phantom(imgInfo_Phantom, 1)
    PhantomC = D2['RF']

    print('\t-----PHANTOM DONE-----')

    class Tree(dict):
        """A tree implementation using python's autovivification feature."""
        def __missing__(self, key):
            value = self[key] = type(self)()
            return value

        #cast a (nested) dict to a (nested) Tree class
        def __init__(self, data={}):
            for k, data in data.items():
                if isinstance(data, dict):
                    self[k] = type(self)(data)
                else:
                    self[k] = data
    '''sms: similar to matlab tables'''

    # remove annoying warnings from output
    if not sys.warnoptions:
        import warnings
        warnings.simplefilter("ignore")

    # does the deed.
    print('PREP DONE. STARTING PROCESSING')
    save_the_whales_videos_2(chosen_IDs,depth,width,ham,hanning,freq_num_for_att)
    '''sms: the core processing'''

#################################################################################################################################
print('Finished DSP, now starting PD process')

## Now ready for PD generation for the whole data-set, and the different Train, Test and Validation-sets

#finalbucket_list = ["liver-data-48x61-alldata","liver-data-24x122-alldata","liver-data-16x183-12x16-alldata","liver-data-48x61-12x16-alldata"]
# Define Pull/Push Bucket
bucket_pull_name = Master_bucket#"liver-data-16x183"
bucket_push_name = Master_bucket#"liver-data-16x183"

bucket_pull = storage_client.get_bucket(bucket_pull_name)
bucket_push = storage_client.get_bucket(bucket_push_name)
#if iterable != 0:
# the data type should not matter, as all videos and frames should be universal across data types. this can be checked with data_checker
caseID , _ , _ , _ , info_list = get_info_frame('rf')

print("Number of Case IDs: "+str(np.shape(caseID)[0]))

# declare dataframes
case_df = pd.DataFrame()
video_df = pd.DataFrame()
#     frame_df = pd.DataFrame()

# list case IDs and names for case level df
case_df['caseID'] = list(caseID)
case_df['name'] = list(caseID)

# list case IDs and names for video level df
video_caseID = []
video_ID = []
for ii in range(0, np.shape(caseID)[0]):
    for jj in range(0, len( info_list[ii][1] )):
        video_caseID.append(caseID[ii])
        video_ID.append(info_list[ii][1][jj])

video_df['caseID'] = video_caseID
video_df['name'] = video_ID

#     # list case IDs and names for frame level df
#     frame_caseID = []
#     frame_ID = []
#     full_name = []
#     for ii in range(0, np.shape(caseID)[0]):
#         for jj in range(0, len( info_list[ii][1] )):
#             # loop through the number of frames for the relevant video
#             for kk in range( 0, info_list[ii][2][jj] ):
#                 frame_caseID.append(caseID[ii])
#                 vid_frame_name = info_list[ii][1][jj] + '_' + str(kk)
#                 frame_ID.append( vid_frame_name )
#                 full_name.append( caseID[ii] + '_' + vid_frame_name )

#     frame_df['caseID'] = frame_caseID
#     frame_df['name'] = frame_ID

print("Fetching the patient demographic data dataframe")
PD_DF1 = pd.read_csv('PD_Nov22_2021_df.csv') #Will be replaced with csv file to be provided by Miriam

# dropping duplicate values
PD_DF1 = PD_DF1.drop_duplicates(subset ="caseID", keep = 'first', inplace = False)#(subset ="CaseID", keep = 'first', inplace = False)

# Get wanted PD data columns
PD_DF = pd.DataFrame()
wanted_data = ['Patient_id', 'Repeated?', 'Site', 'Height', 'Weight',
   'BMI', 'Age', 'Gender', 'History1', 'History2', 'History3',
   'History hbv', 'History hcv', 'History (yes/no)', 'KPA',
   'Fibrosis Grade', 'CAP', 'Steatosis Grade', 'Disease Labelled',
   'Disease', 'Unlabelled Disease', 'Payable', 'irb', 'Ascites',
   'Portal Vein Thrombosis', 'Biopsy', 'Jaundice']#[ 'Disease' , 'History' , 'BMI' , 'FibroscanCAP' , 'FibroscanKPA' , 'Country' , 'Gender' , 'Age' , 'number_of_lesions', 'lesion_size', 'Notes']
PD_DF['caseID'] = PD_DF1['caseID']#PD_DF1['CaseID']
for wanted in wanted_data:
    PD_DF[wanted] = PD_DF1[wanted]

# Find cases that are in both the Dataframe and the processed cases from the bucket
case_df = pd.merge( case_df, PD_DF, how='left', on='caseID' )
video_df = pd.merge( video_df, PD_DF, how='left', on='caseID' )
#     frame_df = pd.merge( frame_df, PD_DF, how='left', on='caseID' )
print( len(case_df) )
print( len(video_df) )
#     print( len(frame_df) )

#####################
#SHADOWING DETECTION
#####################
'''sms: currently not used - moved to step c'''
try:
    count = 0
    #num_cores = multiprocessing.cpu_count()
    shadowing_list = []
    quality_list = []

    for ID in tqdm( caseID ):

        # load the videos
        with parallel_backend('multiprocessing'):
            videos = Parallel(n_jobs=8,mmap_mode='r+')(delayed(load_video)(ID,info_list[count][1][ii],'rf') for ii in range(0,len(info_list[count][1])) )

        # evaluate all frames of each video
        # there is probably no benefit to using multiprocessing here as the video shadowing code is very quick. Multiprocessing might only make it slower in this case

        # multiprocessing
        with parallel_backend('multiprocessing'):
            quality = Parallel(n_jobs=8,mmap_mode='r+')(delayed(video_shadowing_quicker)(videos[ii], 80, 350, 200, 0.08, 0.18) for ii in range(0,len(videos)) )
        # OR inline for loop (slower)
        #quality = [ video_shadowing_quicker(videos[ii], 80, 350, 200, 0.08, 0.18) for ii in range(0,len(videos)) ]

        quality_list.append(quality)

        for ii in range(0,len(videos)):
            shadowing_list.append(quality[ii])

        del videos, quality
        count += 1

        clear_output()
except:
    print("failed first time, will try again")
    count = 0
    #num_cores = multiprocessing.cpu_count()
    shadowing_list = []
    quality_list = []

    for ID in tqdm( caseID ):

        # load the videos
        with parallel_backend('multiprocessing'):
            videos = Parallel(n_jobs=8,mmap_mode='r+')(delayed(load_video)(ID,info_list[count][1][ii],'rf') for ii in range(0,len(info_list[count][1])) )

        # evaluate all frames of each video
        # there is probably no benefit to using multiprocessing here as the video shadowing code is very quick. Multiprocessing might only make it slower in this case

        # multiprocessing
        with parallel_backend('multiprocessing'):
            quality = Parallel(n_jobs=8,mmap_mode='r+')(delayed(video_shadowing_quicker)(videos[ii], 80, 350, 200, 0.08, 0.18) for ii in range(0,len(videos)) )
        # OR inline for loop (slower)
        #quality = [ video_shadowing_quicker(videos[ii], 80, 350, 200, 0.08, 0.18) for ii in range(0,len(videos)) ]

        quality_list.append(quality)

        for ii in range(0,len(videos)):
            shadowing_list.append(quality[ii])

        del videos, quality
        count += 1

        clear_output()

# get one long list for all of the frames
full_shadowing_list = []
for ii in range(0, len(shadowing_list)):
    full_shadowing_list = full_shadowing_list + shadowing_list[ii] 

# APPLY TO DATAFRAME AND SAVE -------------------------------------------------
# remember, a 1 is a PASS, a 0 means too much shadowing

# apply to video dataframe
case_df['shadowing_test'] = quality_list

# apply to video dataframe
video_df['shadowing_test'] = shadowing_list

#     # apply to frames
#     frame_df['shadowing_test'] = full_shadowing_list
#     print(frame_df['shadowing_test'].value_counts())

print( len(case_df) )
print( len(video_df) )
#     print( len(frame_df) )


### Save and upload all processed cases
# save to csv
case_df.to_pickle(Master_bucket+'_dataframes_Nov222021/case_data.pkl')
# upload to bucket
blob = bucket_push.blob( 'case_data.pkl' )
blob.upload_from_filename(Master_bucket+'_dataframes_Nov222021/case_data.pkl' )

# save to csv
video_df.to_pickle(Master_bucket+'_dataframes_Nov222021/video_data.pkl')
# upload to bucket
blob = bucket_push.blob( 'video_data.pkl' )
blob.upload_from_filename(Master_bucket+'_dataframes_Nov222021/video_data.pkl' )

#     # save to csv
#     frame_df.to_pickle(Master_bucket+'_dataframes_Apr122021/frame_data.pkl')
#     # upload to bucket
#     blob = bucket_push.blob( 'frame_data.pkl' )
#     blob.upload_from_filename(Master_bucket+'_dataframes_Apr122021/frame_data.pkl' )

print('Finished generating dataframes. Proceed to Splitting_TrainTest.ipynb')
