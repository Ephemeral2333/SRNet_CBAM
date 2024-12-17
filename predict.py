# predict.py
import torch
from model.SRNet import Model
from PIL import Image
from torchvision import transforms
import config as c

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize model
model = Model().to(device)

# Load pretrained model
if c.pre_trained_srnet_path is not None:
    model.load_state_dict(torch.load(c.pre_trained_srnet_path))
else:
    print('No pre-trained model path provided.')
    exit()

# Define transform
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

def predict(image_path):
    # Load image
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)

    # Predict
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        prediction = outputs.data.max(1)[1]

    return prediction.item()

if __name__ == '__main__':
    image_path = '/root/autodl-tmp/stego/7.pgm'
    print('Predicting image: {:s}'.format(image_path))
    result = predict(image_path)
    if result == 0:
        print('判断为非隐写图像')
    elif result == 1:
        print('判断为隐写图像')