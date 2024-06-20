import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import resnet18
from gradcam import GradCAM
import pytorch_lightning as pl
from torchmetrics.functional import accuracy
import cv2  # OpenCVをインポート



# タイトルとテキストを記入
st.title('犬猫判別アプリケーション')

# 画像の前処理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ネットワークの定義
class ResNetLightning(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature = resnet18(pretrained=True)
        self.fc = nn.Linear(1000, 2)

    def forward(self, x):
        h = self.feature(x)
        h = self.fc(h)
        return h
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
        return optimizer

# 学習済みモデルの読み込み
def load_model(model_path):
    return torch.load(model_path, map_location=torch.device('cpu'))

def predict_image(model, image):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)[0] * 100  # 確率をパーセンテージに変換
        dog_probability = probabilities[1].item()  # 犬の確率
        cat_probability = probabilities[0].item()  # 猫の確率
        
        if dog_probability > cat_probability:
            label = "犬"
            confidence_percentage = dog_probability
        else:
            label = "猫"
            confidence_percentage = cat_probability
        
    return label, confidence_percentage



def apply_gradcam(model, image):
    image_tensor = transform(image).unsqueeze(0)

    gradcam = GradCAM(model.feature, target_layer=model.feature.layer4[-1])  # ResNet18の最終的な畳み込み層を指定

    mask, _ = gradcam(image_tensor)
    heatmap = (mask.detach().numpy().transpose(0, 2, 3, 1)).astype(np.float32)
    heatmap = np.uint8(255 * heatmap)

    # ヒートマップをカラーマップに適用
    heatmap_color = cv2.applyColorMap(255 - heatmap[0], cv2.COLORMAP_JET)

    # 元の画像をNumPy配列に変換
    original_image = np.array(image)
    original_image = cv2.resize(original_image, (224, 224))

    # 元の画像とヒートマップをブレンド
    blended = cv2.addWeighted(original_image, 0.5, heatmap_color, 0.5, 0)

    return blended

# 保存されたモデルの重みを読み込む
weights = torch.load(".\\dog_cat_model_1CNN.pt")
# モデルの重みを読み込む
model = ResNetLightning()
# 重みをモデルにロード
model.load_state_dict(weights)
# モデルを評価モードに設定
model.eval()

# # 画像のアップロード
uploaded_image = st.file_uploader("JPEGファイルをアップロードしてください", type="jpg")
if uploaded_image is not None:
    image = Image.open(uploaded_image)

    # 画像の予測結果を表示
    result, confidence = predict_image(model, image)
    st.write("画像の予測結果:", result)
    st.write("確信度:", int(confidence), "%")

    # Grad-CAMを適用してヒートマップを生成し、表示
    heatmap = apply_gradcam(model, image)
    heatmap = Image.fromarray(heatmap)
    # 2つのカラムを作成
    col1, col2 = st.columns(2)
    # 左のカラムにアップロードした画像を表示
    col1.image(image, caption='アップロードした画像', use_column_width=True)
    # 右のカラムにGrad-CAMの結果を表示
    col2.image(heatmap, caption='Grad-CAMを適応した画像', use_column_width=True)
