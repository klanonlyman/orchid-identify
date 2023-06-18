使用swin-transformer辨識蘭花品種


環境部分: (參考以下網址https://blog.csdn.net/qq_36622589/article/details/117913064) </br>
1.conda create -n swin python=3.7 -y </br>
2.activate swin </br>
3.conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.1 -c pytorch </br>
4.pip install timm==0.3.2 </br>
5.pip install opencv-python==4.4.0.46 termcolor==1.1.0 yacs==0.1.8 </br> 
6.去此網址https://github.com/NVIDIA/apex 下載到本地資料夾，使用CD 進去該資料夾底下 </br>
7.python setup.py install 等待安裝 </br>
8.pip install json </br>

預訓練model: https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth </br>

</br>
使用colab 執行程式碼流程: </br>

1. 上傳訓練資料放入到到名為 data的資料夾 (包含CSV跟圖片) 跟預訓練權重 swin_large_patch4_window12_384_22k.pth
2. 上傳各個.py2檔
3. 執行colab.ipynb照著block 執行
4.最後會生成結果在result的資料夾裡面


不使用colab 執行程式碼流程: </br>

1. 將訓練資料放入到到名為 data的資料夾 (包含CSV跟圖片) 跟預訓練權重 swin_large_patch4_window12_384_22k.pth 放入此目錄底下
2. 執行程式碼 data_augmentation.py
3. 此時會生成 名為data_augmentation的資料夾 (包含CSV跟圖片)
4. 執行程式碼 train_class.py 
5. 會生成名為 train資料夾 裡面的每個資夾分別對應一個label
6. 執行程式碼 train.py 會生成每個epoch的權重
7. 把測試集的資料放到名為 test的資料夾
8. 執行predict.py 去生成對應的submission.csv (這裡預測是拿 55 57 62 的weight)



每個code的功能:</br>
  1.data_augment_tool.py : 各個資料增強的方法 
  2.data_augmentation.py : 資料增強 生成圖片跟對應的CSV檔
  3.train_class.py : 將資料增強的資料分配到對應label的資料夾
  4.model.py : 模組架構
  5.utils.py : 訓練跟資料組裝
  6.my_dataset.py: 讀取資料
  7.train.py : 訓練模型API
  8.predict.py: 預測並生成對應的CSV




  

