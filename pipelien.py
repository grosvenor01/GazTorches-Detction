# pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu
# need the previous installation for using torch with CPU
import torch , pathlib , cv2 ,os
import numpy as np
import pandas as pd
from IPython.display import display
#bonus code (usedfor the relatives path finding)
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

#models loading 
def load_models():
    model = torch.hub.load('Ultralytics/yolov5', 'custom',"best.pt", force_reload=True,trust_repo=True)
    return model

#turning the result into a data frame with the predicted volume 
def stacking_results(model_results , distance):
    df = model_results.pandas().xyxy[0]
    df['width'] = df['xmax'] - df['xmin']
    df['height'] = df['ymax'] - df['ymin']
    df = df.drop(df[df['confidence'] < 0.35].index)
    df['predicted_volume'] = 0
    for _, row in df.iterrows():
        new_distance = distance/10
        predicted_vol = row['height'] * row['width'] * 1
        df['predicted_volume'] = df['predicted_volume'].astype(float)
        df.at[_, 'predicted_volume'] = predicted_vol
    
    return df # returning the full information dataset about the boundings and classes (preview it using print df) 

#build the boundings of a specific class
def boundings_builder(picture_path , df): # build the bounding from the picture and the dataset and return it 
    image = cv2.imread(picture_path)
    boxes = []
    for i in range(len(df)):
        boxes.append([df["xmin"].iloc[i], df["ymin"].iloc[i], df["xmax"].iloc[i], df["ymax"].iloc[i], df["class"].iloc[i], df["predicted_volume"].iloc[i]])
    font = cv2.FONT_HERSHEY_SIMPLEX
    for box in boxes:
        x1, y1, x2, y2, label, percentage = box
        if label == 1:
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (139, 69, 19), 2)
            cv2.putText(image, "gaz-torch", (int(x1), int(y1-10)), font, 0.6, (139, 69, 19), 2, cv2.LINE_AA)
            cv2.putText(image, '',(int(x1), int(y2+20)), font, 0.6, (139, 69, 19), 2, cv2.LINE_AA)
        else :
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            percentage_text = f"{percentage:.2f}Litres"
            cv2.putText(image, percentage_text, (int(x1), int(y2+20)), font, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imwrite('labeled_image2.png', image)
    cv2.destroyAllWindows()

def build_dataset():
    for i in os.listdir("C:/Users/abdo7/Downloads/machine learning"):
        results = model(f"C:/Users/abdo7/Downloads/machine learning/{i}")
        df2  = stacking_results(results , 5)
        df2 = df2[df2["class"]!=0]
        print("avant"+i)
        if(len(df2)>0):
            file  = open("dataset.csv" , 'a')
            file.write(i+','+str(df2.at[df2.index[0],'height'])+','+str(df2.at[df2.index[0],'width'])+','+str(df2.at[df2.index[0],'predicted_volume'])+'\n')
            print(df2.at[df2.index[0],'height'])
            print(i)
#main code
model = load_models()

while(1):
    picture_path = input("give the image path : ")
    distance = int(input("Enter the distance : "))
    results = model(picture_path)
    df = stacking_results(results , distance) 
    boundings_builder(picture_path , df)
