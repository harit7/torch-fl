
from models.lenet import LeNet5
from models.text_bc import TextBinaryClassificationModel

def createModel(config):
    model = None
    arch = config['modelConf']['arch']
    if(arch=='lenet'):
        model = LeNet5()
        
    if(arch=='textBC'):
        model = TextBinaryClassificationModel(config["modelConf"])

    model.to(config['device'])
    return model
 