#KüTÜPHANLER
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import os
#!! "project_directory" değişkeninin kendi uzantınıza göre değiştirmeniz gerekiyor.
project_directory ='D:\\proje' #BU
# CSV dosyasını oku
csv_file_path = project_directory+r"\\annotations.csv"
df = pd.read_csv(csv_file_path)

# HLY sütunundan dosya adlarını ve etiketleri al
file_names = df['HLY']
labels = df['tissue type']

# Dosya seçme butonu
uploaded_file = st.file_uploader("Görüntü Yükleyin", type=["jpg", "jpeg", "png"])
# Model yolları
model_paths = [
    project_directory+"\\model\\modelCNN.h5",
    project_directory+"\\model\\lenet_model.h5",
    project_directory+"\\model\\inception_model.h5",
    project_directory+"\\model\\densenet_model.h5",
    project_directory+"\\model\\EfficientNetB0_model.h5",
]

# Model isimleri
model_names = ["CNN", "LeNet", "GoogLenet", "DenseNet121","EfficientNetB0"]
# Sembol alanı 
selected_model_index = st.sidebar.radio("Model Seçimi", range(len(model_names)), format_func=lambda i: model_names[i])
symbol = model_names[selected_model_index]  # Seçilen modelin adını al
# Model grafiklerini gösterme butonları
if st.sidebar.button("CNN MODELİ GRAFİK"):
    st.title("CNN MODELİ")
    uploaded_file=None  
    #Modeli için dört farklı grafik yolu
    image_path = project_directory+ r"\grafik\cnn.png"  
    image = Image.open(image_path)
    image_path2 = project_directory+ r"\grafik\cnn2.png"  
    image2 = Image.open(image_path2)
    image_path3 = project_directory+ r"\grafik\cnn3.png"  
    image3 = Image.open(image_path3) 
    image_path4 = project_directory+ r"\grafik\cnn4.png"  
    image4 = Image.open(image_path4)
    
    # Grafikleri gösterme
    st.image(image2, use_column_width=True)
    st.image(image3, use_column_width=True)
    st.image(image4, use_column_width=True)
    st.image(image, use_column_width=True, caption="CNN Modeli Confusion Matrix")
    
if st.sidebar.button("LeNet MODELİ GRAFİK"): 
    st.title("LeNet MODELİ")
    uploaded_file=None
    #Modeli için dört farklı grafik yolu
    image_path =project_directory+ r"\grafik\LeNet.png"  
    image = Image.open(image_path)
    image_path2 = project_directory+ r"\grafik\LeNet2.png"  
    image2 = Image.open(image_path2) 
    image_path3 = project_directory+ r"\grafik\LeNet3.png"  
    image3 = Image.open(image_path3) 
    image_path4 = project_directory+ r"\grafik\LeNet4.png"  
    image4 = Image.open(image_path4)
    # Grafikleri gösterme    
    st.image(image2, use_column_width=True)
    st.image(image3, use_column_width=True)
    st.image(image4, use_column_width=True)
    st.image(image, use_column_width=True, caption="LeNet Modeli Confusion Matrix")

if st.sidebar.button("GoogLenet MODELİ GRAFİK"):
    st.title("GoogLenet MODELİ")
    uploaded_file=None
    #Modeli için dört farklı grafik yolu
    image_path = project_directory+ r"\grafik\GoogLenet.png"  
    image = Image.open(image_path)
    image_path2 = project_directory+ r"\grafik\GoogLenet2.png"  
    image2 = Image.open(image_path2)
    image_path3 =project_directory+ r"\grafik\GoogLenet3.png"  
    image3 = Image.open(image_path3)
    image_path4 = project_directory+ r"\grafik\GoogLenet4.png"  
    image4 = Image.open(image_path4)
    # Grafikleri gösterme    
    st.image(image2, use_column_width=True)
    st.image(image3, use_column_width=True)
    st.image(image4, use_column_width=True)
    st.image(image, use_column_width=True, caption="GoogLenet Modeli Confusion Matrix")


if st.sidebar.button("DenseNet121 MODELİ GRAFİK"): 
    st.title("DenseNet121 MODELİ")
    uploaded_file=None
    #Modeli için dört farklı grafik yolu
    image_path = project_directory+ r"\grafik\DenseNet121.png"  
    image = Image.open(image_path)
    image_path2 = project_directory+ r"\grafik\DenseNet121_2.png"  
    image2 = Image.open(image_path2)
    image_path3 = project_directory+ r"\grafik\DenseNet121_3.png"  
    image3 = Image.open(image_path3)
    image_path4 =project_directory+ r"\grafik\DenseNet121_4.png"  
    image4 = Image.open(image_path4)
    # Grafikleri gösterme
    st.image(image2, use_column_width=True)
    st.image(image3, use_column_width=True)
    st.image(image4, use_column_width=True)
    st.image(image, use_column_width=True, caption="DenseNet121 Modeli Confusion Matrix")

if st.sidebar.button("EfficientNetB0 MODELİ GRAFİK"): 
    st.title("EfficientNetB0 MODELİ")
    uploaded_file=None
    #Modeli için dört farklı grafik yolu 
    image_path = project_directory+ r"\grafik\EfficientNetB0.png"  
    image = Image.open(image_path)  
    image_path2 =project_directory+ r"\grafik\EfficientNetB0_2.png"  
    image2 = Image.open(image_path2)   
    image_path3 = project_directory+ r"\grafik\EfficientNetB0_3.png"  
    image3 = Image.open(image_path3)  
    image_path4 = project_directory+ r"\grafik\EfficientNetB0_4.png"  
    image4 = Image.open(image_path4)
    # Grafikleri gösterme
    st.image(image2, use_column_width=True)
    st.image(image3, use_column_width=True)
    st.image(image4, use_column_width=True)
    st.image(image, use_column_width=True, caption="EfficientNetB0 Modeli Confusion Matrix")

#Roc eğrisi Grafiği butonu
if st.sidebar.button("Roc EĞRİSİ"): 
    st.title("Roc EĞRİSİ")
    uploaded_file=None
    image_path = project_directory+ r"\grafik\roc.png"  
    image = Image.open(image_path)
    st.image(image, caption='ROC Curve', use_column_width=True)
#Sayfa içeriğini temizleme butonu
if st.sidebar.button("TEMİZLE"):
    uploaded_file=None
    pass

# Seçilen modeli yükle
loaded_model = load_model(model_paths[selected_model_index])
# Yüklenen dosya varsa
if uploaded_file is not None:
    st.title(f"**{symbol} - AI Modelinin Tahmin Sonucu:**")
    # Gerçek etiket değerini gösterme
    selected_file_name = uploaded_file.name
    actual_label = labels[file_names[file_names == selected_file_name].index[0]]
    image = Image.open(uploaded_file)
    image = image.resize((200, 200))
    # Giriş şeklini modelin beklentisine uygun şekilde yeniden şekillendirme   
    if actual_label == 'LGC'or actual_label == 'NST':
        image_array = np.array(image)
        image_array = image_array / 255.0
        # Giriş şeklini modelin beklentisine uygun şekilde yeniden şekillendirme
        image_array = image_array.reshape((1, 200, 200, 3))
    else:
        image_array = np.array(image)
        # Giriş şeklini modelin beklentisine uygun şekilde yeniden şekillendirme
        image_array = image_array.reshape((1, 200, 200, 3))
    
    # Tahmin yapma
    result = loaded_model.predict(image_array)

    # Streamlit düzeni
    col1, col2= st.columns([6,6])  # Kolonları ayarla
    
    # Sol tarafta görüntü 
    col1.write(f"**Gerçek Etiket Değeri:** {actual_label}")
    col1.image(image, use_column_width=True)
    selected_model_name = model_names[selected_model_index]
    
    # Tahmin edilen etiketi yazdırma
    predicted_label_index = np.argmax(result)
    predicted_label = labels.unique()[predicted_label_index]
    col2.write(f"**Tahmin Edilen Etiket:** {predicted_label}")
    
    # Tahminleri görselleştirme
    fig, ax = plt.subplots()
    ax.bar(labels.unique(), result[0])
    ax.set_ylabel('Probability')
    ax.set_title('Predictions')
    col2.pyplot(fig)
    

# Tahmin edilen etiketlerin yüzde oranlarını yazdırma
    etiket_yuzde_oranlari = {label: f"%{result[0][i]*100:.1f}" for i, label in enumerate(labels.unique())}
    etiket_yazi = "\n".join([f"{label}:{oran} " for label, oran in etiket_yuzde_oranlari.items()])
    col2.write("Tahmin Edilen Etiketler:")
    col2.write(etiket_yazi)
    
    
