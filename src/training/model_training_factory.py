from training.text_bc_training import TextBCModelTraining
from training.model_training import ModelTraining

def getModelTrainer(conf,isAttacker=False):
    model = conf['arch']
    if(model=='textBC'):
        return TextBCModelTraining(conf,isAttacker=isAttacker)
    if(model=='lenet5'):
        return ModelTraining(conf,isAttacker=isAttacker)
    
