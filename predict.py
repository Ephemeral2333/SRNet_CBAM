# predict.py
import torch
import argparse
from model.SRNet_CBAM import Model
from PIL import Image
from torchvision import transforms
import config as c

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

args = argparse.ArgumentParser(description='SRNet')
args.add_argument('--image_path', type=str, help='image path')
args.add_argument('--pretrained_path', type=str, default=None, help='pre-trained model path')
args = args.parse_args()

# Initialize model
model = Model().to(device)

# Load pretrained model
if args.pretrained_path:
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
    image_path = args.image_path
    if image_path is None:
        print('No image path provided.')
        exit()
    result = predict(image_path)
    if result == 0:
        print('Judged as cover image')
    elif result == 1:
        print('Judged as stego image')