import sys
import argparse

if sys.version_info[0] > 2:
    import configparser as ConfParser
else:
    import ConfigParser as ConfParser

import os
import glob
import shutil
from subprocess import call


def set_config():
    # Anima config
    configFilePath = os.path.join(os.path.expanduser("~"), ".anima",  "config.txt")
    if not os.path.exists(configFilePath):
        print('Please create a configuration file for Anima python scripts. Refer to the README')
        quit()

    configParser = ConfParser.RawConfigParser()
    configParser.read(configFilePath)

    animaDir = configParser.get("anima-scripts", 'anima')
    animaDataDir = configParser.get("anima-scripts", 'extra-data-root')
    animaScriptsDir = configParser.get("anima-scripts", 'anima-scripts-public-root')
    animaPrivScriptsDir = configParser.get("anima-scripts", 'anima-scripts-root')

    # TractSeg config
    configFilePath = os.path.join(os.path.expanduser("~"), ".tractseg-config",  "config.txt")
    if not os.path.exists(configFilePath):
        print('Please create a configuration file for tractseg.')
        quit()

    configParser = ConfParser.RawConfigParser()
    configParser.read(configFilePath)

    tractsegDir = configParser.get("tractseg", 'tractseg-bin')
    mrtrixDir = configParser.get("tractseg", 'mrtrix-bin')

    config = {
        'animaDir': animaDir,
        'animaDataDir': animaDataDir,
        'animaScriptsDir': animaScriptsDir,
        'animaPrivScriptsDir': animaPrivScriptsDir,
        'tractsegDir': tractsegDir,
        'mrtrixDir': mrtrixDir
    }

    animaConvertImage = os.path.join(animaDir, "animaConvertImage")
    animaApplyTransformSerie = os.path.join(animaDir, "animaApplyTransformSerie")
    animaTensorApplyTransformSerie = os.path.join(animaDir, "animaTensorApplyTransformSerie")
    animaMCMApplyTransformSerie = os.path.join(animaDir, "animaMCMApplyTransformSerie")
    animaCropImage = os.path.join(animaDir, "animaCropImage")
    animaGMMT2RelaxometryEstimation = os.path.join(animaDir, "animaGMMT2RelaxometryEstimation")
    animaDTIScalarMaps = os.path.join(animaDir, "animaDTIScalarMaps")
    animaMCMScalarMaps = os.path.join(animaDir, "animaMCMScalarMaps")

    animaBrainExtraction = os.path.join(animaScriptsDir,"brain_extraction","animaAtlasBasedBrainExtraction.py")
    myAnimaSubjectsMCMFiberPreparation = os.path.join(animaPrivScriptsDir, "diffusion", "mcm_fiber_atlas_comparison", "myAnimaSubjectsMCMFiberPreparation.py")
    animaRegister3DImageOnAtlas = os.path.join(animaScriptsDir, "registration", "animaRegister3DImageOnAtlas.py")
    antsRegister3DImageOnAtlas = os.path.join(animaPrivScriptsDir, "registration", "antsRegister3DImageOnAtlas.py")
    animaDiffusionImagePreprocessing = os.path.join(animaScriptsDir, "diffusion", "animaDiffusionImagePreprocessing.py")
    Tractometry = os.path.join(animaPrivScriptsDir, "tractometry_Julie.py")
    
    #Check if fast is in path
    fast = shutil.which("fast")
    if fast is not None:
        fast = ['fast']
    else:
        #Requires docker
        fast = ['fsl','fast']

    tools = {
        'animaConvertImage': animaConvertImage,
        'animaApplyTransformSerie': animaApplyTransformSerie,
        'animaTensorApplyTransformSerie': animaTensorApplyTransformSerie,
        'animaMCMApplyTransformSerie': animaMCMApplyTransformSerie,
        'animaCropImage': animaCropImage,
        'animaGMMT2RelaxometryEstimation': animaGMMT2RelaxometryEstimation,
        'animaDTIScalarMaps': animaDTIScalarMaps,
        'animaMCMScalarMaps': animaMCMScalarMaps,
        'animaBrainExtraction': animaBrainExtraction,
        'myAnimaSubjectsMCMFiberPreparation': myAnimaSubjectsMCMFiberPreparation,
        'animaRegister3DImageOnAtlas': animaRegister3DImageOnAtlas,
        'antsRegister3DImageOnAtlas': antsRegister3DImageOnAtlas,
        'Tractometry': Tractometry,
        'animaDiffusionImagePreprocessing': animaDiffusionImagePreprocessing,
        'fast': fast
    }

    return config, tools


if __name__ == '__main__':
    set_config()    
    