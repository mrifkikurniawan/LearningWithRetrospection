from easydict import EasyDict as edict

__all__ = ["initialize_dataset", "initialize_loss", "initialize_optimizer"]

def initialize_loss(module: object, loss: str, **args):
    loss_name = loss
    print(f"initializing loss function: {loss_name}")
    loss_ = getattr(module, loss_name)
    loss_ = loss_(**args)
    return loss_

def initialize_dataset(module: object, dataset: str, **args):
    from torchvision import transforms
    
    print(f"initializing dataset {dataset}")

    args = edict(args)
    transform = eval(args.transform)
    args.transform = transform
    
    dataset_ = getattr(module, dataset)
    dataset_ = dataset_(**args)
    return dataset_

def initialize_optimizer(module: object, method: str):
    print(f"Initializing optimizer {method}")

    optimizer = getattr(module, method)
    return optimizer