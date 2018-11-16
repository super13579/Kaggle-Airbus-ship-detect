# Kaggle-Airbus-ship-detect
Kaggle- AirBus Ship detect (Unet)

This is Kaggle Airbus Ship detection Chanllenge  

![image](https://github.com/super13579/Kaggle-Airbus-ship-detect/blob/master/ship_detect.JPG)   

Share my experience of this competitions (But I don't get a good grade...)

Step 1 ==> Ship exist detection by Transfer learning of ResNet50  
https://www.kaggle.com/super13579/simple-transfer-learning-detect-ship-exist-keras  
Step 2 ==> ResNet34 + Unet to do Image Segmentation  
https://www.kaggle.com/super13579/u-net-base-on-resnet34-transfer-learning-keras/notebook  
Step 3 ==> Use Step 1 to separate Non_ship and ship then do Step 2 on Ship Images  
https://www.kaggle.com/super13579/unet34-predict-result  


My Model Result:  

![image](https://github.com/super13579/Kaggle-Airbus-ship-detect/blob/master/Result_1.JPG)  
![image](https://github.com/super13579/Kaggle-Airbus-ship-detect/blob/master/Result_2.JPG)  
