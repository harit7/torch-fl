from training.text_bc_training import TextBCModelTraining
from training.model_training import ModelTraining

def getModelTrainer(conf):
    model = conf['arch']
    if(model=='textBC'):
        return TextBCModelTraining(conf)
    if(model=='lenet5'):
        return ModelTraining(conf)
    
