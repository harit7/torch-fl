from training.text_bc_training import TextBCModelTraining
from training.generic_model_training import GenericModelTraining

def getModelTrainer(conf,isAttacker=False):
    model = conf['arch']
    if(model=='textBC'):
        return TextBCModelTraining(conf,isAttacker=isAttacker)
    if(model=='lenet5'):
        return GenericModelTraining(conf,isAttacker=isAttacker)
    
