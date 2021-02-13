* 問題1: 怎麼從多層資料夾讀進檔案（numpy 幾x幾的格式）
  * 需要時再計算指定path讀取
* 問題2: 有些frame沒有 bboxes/landmarks 如何對齊
  * 承上題: 如果檔案path不存在就回傳[]
* 問題3: tarfile如何直接讀取到變數裡而不要先佔空間
        tar = tarfile.open(...)
        members = tar.getmembers()
        ...
* 問題4: Speed up
  * https://ai.googleblog.com/2020/05/speeding-up-neural-network-training.html
  * 3-1: 讀資料ram不夠
    * ram需求:影片數x偵數x長x寬x通道數x4bytes 一定爆
    * **重新設計演算法**，不能一次全讀
      * 每次擷取指定影片指定frame or 全部圖片擷取後傳path
  * 3-2: 讀資料太慢
    1. 讀影片用什麼?
      * 哪個快？cv2與pyAV與torchvision
      * 哪個快？每次擷取指定影片指定frame >> 全部圖片擷取後傳path (cv2>>PIL) 
        * **get_pic_time: 100張 3>>0.34>>0.008s, PIL is the best**
        * **PIL還要加上擷取圖片時間的trade-off:1400張兩分鐘** (不用考慮空間，1M圖片才36G)
        * **6epoch 1million pics:130000<<240000s, video_frame比較快（大概第三個epoch以後）** 
    2. 怎麼變成numpy
      * 哪個快？直接讀成numpy或是先讀再變成numpy
        * 開發時間vs讀取時間
    3. 讀圖片用什麼？
      * **PIL-Simd >>PIL >>cv2**
      * cv2轉成RGB圖片後，用cv2 or PIL read都沒問題，但用plt讀會變成BGR

  
* 問題5: bounding box的座標與label probability怎麼同時預測(尺度不同)
  1. 用不同的model, 再用一個Model把全部models包起來return各自的標的
  2. Yolo方法
