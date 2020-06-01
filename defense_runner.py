import os
import sys
sys.path.append('./src/')
sys.path.append('./src/fl-train/')
sys.path.append('./src/training/')
sys.path.append('./src/dataset/*')
import argparse
from fl_train import FLTrainer
from globalUtils import *
import copy


from multiprocessing import Process

def getConfParamVal(key,conf):
    keys = key.split('.')
    y = conf
    for k in keys:
        y = y[k]
    return y

def runConf(conf):
    print('running conf with defense '+conf['defenseTechnique'])
    seed(42)
    od = conf['outputDir']
    if(od.startswith('$')):
        od = od[1:]
        x = od.split('/')
        name = '_'.join( [ '{}_{}'.format(k, getConfParamVal(k,conf)) for k in x[-1].split('_') ] )
        od= '/'.join(x[:-1])+'/'+name
        conf['outputDir'] = od

    if not os.path.exists(od):
        os.makedirs(od)
    print('Will be saving output and logs in directory {}'.format(od))

    stdoutFlag = False
    logger = getLogger("{}/fl.log".format(od), stdoutFlag, logging.INFO)
    print('Log File: {}/fl.log'.format(od))
    #print(conf['ckptEpochs'])

    flTrainer = FLTrainer(conf,logger)
    flTrainer.trainNEpochs()

import time


def ParallelRun(lstConf):
    lstP = []
    for conf in lstConf:
        conf = copy.deepcopy(conf) # ensure no shit happens
        p = Process(target = runConf, args=(conf,))
        p.start()
        lstP.append(p)
        
    for p in lstP:
        p.join()
        
def SeqRun(lstConf):
    for conf in lstConf:
        runConf(conf)
     
     
'''
def yourfunction(arg):
    for i in range(0,100):
        print(str(arg)+'_'+str(i))
        time.sleep(2)

if __name__ == '__main__':
    lst = [1,2,3,4,5]
    lstP = []
    for l in lst:
        p =  Process(target=yourfunction, args=(l,))
        print('x')
        p.start()
        print('y')
        lstP.append(p)
       
    for p in lstP:
        p.join()
        print('z')
    

'''

if __name__ == "__main__":

  
    parser = argparse.ArgumentParser(description='Process args')
    parser.add_argument('--config', type=str,
                        help='The conf file to be used for the training')

    args = parser.parse_args()
    
    
    confFilePath = args.config
    conf0 = loadConfig(confFilePath)
    
    defenses = ['noDefense','normClipping','krum','multiKrum','rfa']
    normBounds = {'noDefense':10, 'normClipping':1.5, 'krum':1.5,'multiKrum':1.5,'rfa':1.5}
    
    lstConf = []
    
    par = True
    conf0['attackFromEpoch']=100
    conf0['numFLEpochs'] = 300
    conf0['enableCkpt'] = False
    
    op_pfx = './outputs/yorgos_backdoor_defenses'
    
    for defense in defenses:
        conf = copy.deepcopy(conf0)
        conf['defenseTechnique'] = defense
        conf['normBound'] = normBounds[defense]
        conf['outputDir'] = op_pfx+ '_'+defense+'_'+str(normBounds[defense])+'/'
        lstConf.append(conf)
    if(par):
        ParallelRun(lstConf)
    else:
        SeqRun(lstConf)
        
    print('Here')
          
        
    
