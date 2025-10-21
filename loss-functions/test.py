import torchvision
import os

model_dir = "./model" 
model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(pretrained=False)
weights_path = os.path.join(model_dir, "maskrcnn_resnet50_fpn_v2.pt")
                
print("Model loaded successfully:", model)