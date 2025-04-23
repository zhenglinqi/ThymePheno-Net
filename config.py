import torch
import os

class Config:

    data_root = "/root/autodl-tmp/thyme_raw_data_4pheno"


    batch_size = 32
    num_epochs = 100
    learning_rate = 0.001


    num_classes = 4  


    device = "cuda" if torch.cuda.is_available() else "cpu"


    num_workers = 2
    prefetch_factor = 2


    model_save_path = "/root/autodl-tmp/saved_models_4pheno/DApheno_Resnet111"
    os.makedirs(model_save_path, exist_ok=True)


    mixed_precision = True
    gradient_accumulation_steps = 1  
    weight_decay = 0.01  
    warmup_epochs = 5  
    patience = 10  
    min_lr = 1e-6 


    img_size = 224
    train_transforms_params = {
        'scale': (0.8, 1.0),
        'rotation': 30,
        'color_jitter': {
            'brightness': 0.2,
            'contrast': 0.2,
            'saturation': 0.2
        }
    }
