Dataset, DataLoader詳解

https://www.cnblogs.com/marsggbo/p/11308889.html

- 調適dataset

* .\_\_getitem__()

* 你可以写一个for循环遍历dataset来进行调试了

* 一般return x, y

* 可以return dict, 在dataloader時用key來取就好

* 可以return x1, x2, ..., y, 在dataloader時用同樣的數量來接就好

* 注意x每個樣本的形狀要一樣(尤其圖片),x1x2...都是（所以要resize and crop）

* 如果只有特定label有缺data, 如bboxes or landmarks那不要用跳過recursion的方式，因為dataloader在shuffle時會出問題（maximum recursion）

* float16的精度沒辦法儲存五位數以上, 所以10071~10079都會被儲存為1.007e4+04, 這時請用float32

* 請檢查data eg.

* 每個image frame都真的是正常圖片？

* image frame跟label有沒有對起來

* bboxes, landmarks是同一個idx缺, 還是各自缺

* data imbalance問題

- transformation

* 用在PIL圖片上

* 若要用在boundingbox or landmarks 需客製

* 注意記憶體用量, skimage.transform.resize會吃太多記憶體

- dataloader

* sampler取樣機制

* .collate_fn()決定如何打包

* 如果回傳的不只img, label則可能要覆寫

* num_workers會讓每個worker的dataset物件都不同, 所以存取成員時也不會取到一樣的

plt.subplot

* ax = plt.subplot(1, total_nums, now_num)

train

- 多個東西要計算loss怎麼辦？

* YOLO3一個一個分別計算再加總再backward

loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])

loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])

loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])

loss_h = self.mse_loss(h[obj_mask], th[obj_mask])

loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])

loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])

loss_conf = self.obj_scale ** loss_conf_obj + self.noobj_scale ** loss_conf_noobj

loss_cls = self.bce_loss(pred_cls[obj_mask], tcls[obj_mask])

total_loss = loss_x + loss_y + loss_w + loss_h + loss_conf + loss_cls

return output, total_loss

deploy

* client, server的data交互

* server.py

@app.route('/predict', method=['POST'])

def predict():

if request.method == 'POST'

file = request.files['file']

image_byte = file.read()

#transform and predict

return jsonify({'probability': prob, 'class_name':class_name})

* client.py

response = request.post('.../predict',

files = {'file', open('.../123.jpg','rb')})

* read as byte再用is.ByteReader處理

* server.py

...

image = Image.open(io.BytesIO(image_bytes))

* client.py同上

[1. Pipleline Practice](#1)

**1. Pipeline Practice**<a name='1'></a>
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

* deploy需要
  * https://github.com/musikalkemist/Deep-Learning-Audio-Application-From-Design-to-Deployment
  1. 決定如何deploy
    * online:flask api/uWSGI/docker+nginx/AWS
    * edge
  2. 基本需要
    * prepare_dataset.py
    * train.py:load_data/prepare_dataset/build_model/train/plot_history/main
    * server.py: def predict function in flask
    * client.py: open data->send data to server side->get result from server

            response = request.post(data)
  3. edge
  * https://nanonets.com/blog/how-to-easily-detect-objects-with-deep-learning-on-raspberry-pi/
  * https://www.rs-online.com/designspark/google-teachable-machine-raspberry-pi-4-cn
    * 縮小model(quantize model)
    * Install TensorFlow on the Raspberry Pi
    * 用電腦操作raspberry pi
      * 拍照/錄影->預測

linux
* pwd
* cd
* mkdir
* ls
* cat

os
* os.walk(path)
  * return generator: yield tuple(root, folders, files)
* os.listdir(path)
* os.path.join()
  * ''.join((a,b,c))
* os.path.exists(path)
* os.path.dirname(path)
* os.path.basename(path)
* os.getcwd()
* os.chdir(path) 
* os.mkdir(path)
* os.remove(file_path)
* shutil.rmtree(folder_path)
* data->model->deploy
    
python
* sorted vs list.sort()
  * sorted return a new list while list.sort() operates inplace
  * when using list, list.sort is faster
* list to tuple
  * tuple(list_name)
* string list to int list
  * list(map(int, list_name))
* 改變形狀用np.reshape()即可
* cv2.VideoCapture().get(number)
  * 1:現在偵數
  * 3:寬度
  * 4:高度
  * 5:偵率
  * 7:總偵數
