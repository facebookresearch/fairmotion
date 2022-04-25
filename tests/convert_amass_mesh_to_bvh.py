import torch
import numpy as np
from human_body_prior.body_model.body_model import BodyModel
from fairmotion.core import motion as motion_class
from fairmotion.ops import conversions
from fairmotion.utils import 
from fairmotion.data import amass,bvh

def load_body_model(bm_path, num_betas=10, model_type="smplh"):
    comp_device = torch.device("cpu")
    bm = BodyModel(
        bm_fname=bm_path, 
        num_betas=num_betas, 
        # model_type=model_type
    ).to(comp_device)
    return bm
  
  file='./*.npz'#TODO --- file from amass
  bm_path = '../smplh/male/model.npz'
  num_betas=10 #TODO --- set tp 16 with hand but this tool doesn't support hand
  model_type='smplh'
  bm = load_body_model(bm_path,num_betas,model_type)
  
  motion = amass.load(file,bm,bm_path,model_type)
  bvh.save(motion,filename='./result.bvh')
  
