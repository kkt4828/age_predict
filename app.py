import streamlit as st
from model.model import ConvNet
# import torch.nn as nn
import torch
# from io import StringIO
from PIL import Image
from torchvision.transforms import ToTensor
# from torchvision import models

st.title('Hello Streamlit')
st.header('Start!!', divider='rainbow')

# number = st.number_input('Input your age', value=5)
# number = float(number)
# model = SimpleNet(1, 1)
# num = model(torch.tensor([number]))
# st.write('Your age : ', num.item())
tf_tensor = ToTensor()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ConvNet()
st.subheader('얼굴 나이 계산기')
number = st.number_input("실제나이를 입력해주세요", value=20, placeholder='숫자 입력')
uploaded_file = st.file_uploader('Insert your face')
if uploaded_file is not None:

    img = Image.open(uploaded_file)
    st.image(img)
    img = img.resize(size=(224, 224))
    img = tf_tensor(img)
    img = img.unsqueeze(0)
    img = model(img)
    st.subheader(f"실제 나이 : {number}세")
    st.subheader(f"AI가 예측한 나이 : {img.item():.2f}세")
