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




Dataset, DataLoader詳解
https://www.cnblogs.com/marsggbo/p/11308889.html
* 調適dataset
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
      * image frame跟label有沒有對起來
      * bboxes, landmarks是同一個idx缺, 還是各自缺
      * data imbalance問題

* transformation
  * 用在PIL圖片上
  * 若要用在boundingbox or landmarks 需客製
    * 注意記憶體用量, skimage.transform.resize會吃太多記憶體
* dataloader
  * sampler取樣機制
  * .collate_fn()決定如何打包
    * 如果回傳的不只img, label則可能要覆寫
  * num_workers會讓每個worker的dataset物件都不同, 所以存取成員時也不會取到一樣的
    
plt.subplot
  * ax = plt.subplot(1, total_nums, now_num)

train
* 多個東西要計算loss怎麼辦？
  * YOLO3一個一個分別計算再加總再backward
          loss_x = self.mse_loss(x[obj_mask], tx[obj_mask])
          loss_y = self.mse_loss(y[obj_mask], ty[obj_mask])
          loss_w = self.mse_loss(w[obj_mask], tw[obj_mask])
          loss_h = self.mse_loss(h[obj_mask], th[obj_mask])
          loss_conf_obj = self.bce_loss(pred_conf[obj_mask], tconf[obj_mask])
          loss_conf_noobj = self.bce_loss(pred_conf[noobj_mask], tconf[noobj_mask])
          loss_conf = self.obj_scale * loss_conf_obj + self.noobj_scale * loss_conf_noobj
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






#Acitivate function
  * sigmoid
    * 將值映射到0-1
    * 問題：
      * 太容易飽和(saturate): 稍微往左右求導就趨近於0
      * gradient vanishing: 導數範圍:0~0.25, 當層數變多時就會出現梯度消失的問題
        * 往回傳播個幾層時太早就變很小, 所以最前面的layer很難更新, 然後後面的雖然更新很大, 但是還是從前面學習過來, 所以沒有用
  * tanh
    * 將值映射到-1~1
    * center到0, 所以比sigmoid收斂更快
      * 未證實：想像sigmoid輸出都是正數, 所以在參數更新的時候通常所有參數都是同一個方向, 只能所有參數一起鋸齒狀更新zigzag, 所以比較慢
    * 比sigmoid有稍微更大的導數範圍(0~1)
    * 只是稍微緩解sigmoid的問題

  * ReLU(rectified linear unit) vs sigmoid vs tanh
    * ReLU適用於cnn模型
    * 解決gradient vanishing, 導數為0 or 1, 不會有消失問題
    * ReLU可以降低overfit：讓某些神經元輸出為0, 等於關閉某些神經元, 變成稀疏神經網路
      * **缺點**：關閉後難以再開啟（因為會被關閉就是因為前面對應的神經元計算後在此產生<0的值(bias太負), 但關閉後代表不會backprob回去不會更新, 下次計算到此神經元還是<0, 還是關閉）
      * **解決**：LReLU PReLU CReLU ELU SELU

  * softmax
    * 是泛化的sigmoid
      * sigmoid針對兩項之總機率=1
      * softmax針對多項之總機率=1
    * 如果要multi-label, 則要用sigmoid預測每個class是/否的機率, 不可用softmax, 因為softmax
      
  

#Loss
###regression
  1. MSE
    * MSE = Mean(SSE) in 統計
    * 若用於logistic regresseion時
      * sigmoid輸出值接近0 or 1時, 因微分後的公式, **梯度會接近0**, 更新會太慢
    * **RMSE**用於跟MAE同range比較, outlier大時會大MAE很多
  2. MAE
    * 微分會有問題：
$\ f(x) = |x|$ 則 $f'(x) = \frac{x}{|x|} $  
      1. 在x=0處不連續
      2. 微分/梯度始終相同(方向可能不同)
        * 對outlier or pred差較遠者懲罰少學習慢
        * pred較近者又太大(可能會在底部跳來跳去            
  * 其實outlier在真實資料中佔很少, 所以對於outlier對於DL的影響其實不大?, 兩者使用差異反而主要在L1, L2 norm

###classification
  0. multi-class vs multi-label
    * multi-class: 很多class但最終預測只有一個class(1,0,0), eg.動物分類器, MNIST數字分類器
    * mulyi-label: 很多class,最終預測多個class(1,1,0), eg. 電影分類(可能同時是科幻, 劇情, 動作)
  1. classification loss
    * only right or wrong
    * 0.1, 0.9 vs 0.4, 0.6 的loss都是一樣的, 但很明顯後者需要改變
    * cant backprob?
  2. 0/1 loss
    * $loss_i = 0$ only when $y_{i,true} = y_{i,pred}$, 太嚴格
    * not convex, 難以優化
  3. perceptron loss
    * 0/1 loss 的改進版
    * $loss_i = 0$ when
    * 有另一版本是hinge loss的變種?
$|y_{i,true}-y_{i,pred}|>t,\ t$為thread value 
  4. cross entropy
    * information gain
      * 公式：
$\ I(x) = -log(p_i)$
      * 越確定的訊息量越低
$p = 100\%$
    * entropy
      * 公式： 
$\ H(X) = \sum_{i=0}^{N}-p_ilog(p_i)$
      * 測量不確定性
      * 越不確定越隨機($p = 50\\%$), entropy越高
    * 公式（有些人有除以N）： 
$\ L(X)=H(X) = \sum_{i=1}^{N}\sum_{c=1}^{C}-y_{true,i,c}log(p_{i,c})$
      * ***二分類問題：***
$y_{true,i}= 0\ or\ 1\ for\ label_1\ or\ label_2\\ y_{pred,i,1},y_{pred,i,2}=(p_1, p_2),\ 其中p_1+p_2=1\\L(X)=\sum_{i=1}^{N}ylog(a)+(1-y)log(1-a),\\其中y=y_{true,i}\ and\ a=y_{pred,i,1}\ and\ (1-a)=y_{pred,i,2}$ 
        * 即為**Binary CE**, 也可運用在多分類/multi-label問題(每個樣本i對每個類別c作一次BCE)https://clay-atlas.com/blog/2019/12/18/machine-learning-chinese-pytorch-tutorial-binary-cross-entroy-loss/
        * 通常用**BCE + Sigmoid**

      * ***多分類問題：***可以簡化, 因為 **multi-classification** 中y_true向量為OHE, 其中只會有一個類別=1, 其餘=0
        * **log loss** 即為簡化後之結果：
$\ H(X) = \sum_{i=1}^{N}-log(p_{i})$
          * 通常用於logistic regression
    * 應用
      * categorical vs sparse CE
        * 其實loss func都一樣, pred也都是prob vector, 差別在label的格式不一樣:OHE or int-encoded
        * **categorical CE** is for OHE Label, eg. [1, 0, 0], [0, 0, 1]
        * **sparese CE** is for int label, eg. [1], [3], [2]
          * 其實只是運算過程簡化, 直接取出label對應的pred vector element
          * when you have many classes (and samples), in which case a **one-hot encoding can be significantly more wasteful** than a simple integer 
          * integer encoding is more compact than one-hot encoding and thus more suitable for encoding sparse binary data.
          * only suitable for "sparse labels", where original vector has exactly one value is 1 and all others are 0
      * sigmoid vs softmax
        * sigmoid is for 二元分類, softmax 則是泛化於多分類multi-classification
        * why sigmoid+BCE 用於二分類：  
$y_{pred,i,1} = \frac{e^{-\beta*X}}{1+e^{-\beta*X}}\ \ (1)\\y_{pred,i,2} = \frac{1}{1+e^{-\beta*X}}\ \ (2)\\y_{pred,i,1}+y_{pred,i,2} = 1\ \ (1)+(2)$
        * 承上, 可推 softmax+CE 用於多分類
      * multi-label 
        * 要用BCE而不是Categorical CE, 因為後者只計算唯一個標籤值為1的狀況
        * 也可用Sigmoid activation + normal CE

  5. hinge loss
    * 公式：
$loss_i = max(0, 1-y_{true,i}*y_{pred,i}),其中\\y_{true,i}=-1\ or\ 1$
    * 並不鼓勵分類器太過自信:让某个正确分类的样本距离分割线超过1并不会有任何奖励，从而使分类器可以更专注于整体的误差。  
$if\ y_{true,i}, y_{pred,i} = 1, 100\\=>\ max(0, 1-1*100)\ is\ still\ 0$
    * 用於二分類, SVM, 間隔最大化問題

###改進
https://chih-sheng-huang821.medium.com/%E6%A9%9F%E5%99%A8-%E6%B7%B1%E5%BA%A6%E5%AD%B8%E7%BF%92-%E6%90%8D%E5%A4%B1%E5%87%BD%E6%95%B8-loss-function-huber-loss%E5%92%8C-focal-loss-bb757494f85e
  1. Huber loss
    * MSE/MAE的合成 to solve MSE/MAE problem
      * for MSE: 對outlier down-weighting
      * for MAE: 對閾值內的採用MSE使梯度有大小
    * 公式：
$\ loss_i = \frac{1}{2}(y_{true,i}-y_{pred,i})^2,\ |y_{true,i}-y_{pred,i}|<\delta\\loss_i=\delta(|y_{true,i}-y_{pred,i}|-\frac{1}{2}\delta),\ O.W.$ 

  2. Focal loss
    * solve 物件偵測easy/hard sample數量差異過大, 造成loss極度imbalance, 學習太偏。 希望主要能學習hard example(前景物件)
      * 候選物件或anchor絕大比例(1000:1)為背景而非前景(物件), 訓練的時候total loss是所有候選物件的loss相加, 這時候1000筆easy examples的loss相加絕對會大於1個hard example的loss, 所以模型主要都在學習背景
      * 對inliers(easy example) down-weighting
    * 公式：
$FL_i(p_i)=-\alpha(1-p_i)^rlog(p_i)$
      * $\alpha:\ \alpha$-balance 可以提高一點正確率
      * $(1-p_i)^r:\ $ modulating factor
      * 當α=1且r=0，focal loss=cross entropy
  3. Custom Loss
    * eg. bounding box loss: SmoothL1/IoU/GIoU/DIoU/CIoU...
      * 因為MSE假設bbox的x, y, w, h都是獨立的, 但事實上應該不是, 所以特製



#Design Process
https://www.youtube.com/watch?v=g2vlqhefADk

###1. 搜尋符合現有task的architecture
  * **transfer learning**: using model that pre-trained on large open dataset and fine-tune

  * 如果需要special case, things that not showed up before, **THEN**

###2. 搜尋好用的design patterns, eg.
**(1) for CV**
  * **efficient model architectures**
    * eg. EfficientNet, ShuffleNet, MobileNet...
    * seperable depthwise conv 
      * 減少計算量、參數量
      * 最後使用1x1conv來學習跨通道特徵
      * 舉例：input(32x32x256), output(32,32,128)
        * 原本要有128個3x3x256kernel做計算
          * 參數量(少算bias):3x3x256x128
          * 計算量(少算加法與bias):32x32x3x3x256x128
        * 現在只要先用256個3x3x1kernel分別對256channel作卷積後(不sum)輸出256張map, 再用128個1x1x256的conv來獲取跨通道特徵, 輸出128張feature map
          * 參數量(少算bias):256x3x3+128x1x1x256
          * 計算量(少算加法與bias):32x32x3x3

    **Some Practical Guidelines in paper: ShuffleNetV2**
    1. 在一個conv layer的兩端用equal channel width的feature map可以minimize memory access cost
      * **not to change input/output channels of a layer too frequently**
      * 當然也有打破規則的, eg. SqueezeNet, MobileNetV2
    2. extremely narrow bottlenecks hurt training stability
      * bottleneck: layers having very few filters(<=8), 通常是指為減少計算量而使用1x1conv先來降維的中間層
      * a few dead neurons and the entire model collapses
      * **using linear activation instead of ReLU after previous conv layer**
    3. network fragmentation reduces the degree of parallelism
      * **using a few large operations is better than a lot of small ones**
      * since the later one is not very GPU-friendly and introduce extra overhead(額外付出的時間)
    4. point-wise operations eg.ReLU and 1x1Conv are not free
      * Memory access cost is non-negligibal although small number of fp operations

  * **inception network**
    * 獲得不同空間尺度特徵(4條路線), 並在加深的過程降低計算量
      * 不同空間尺度： 1x1, 3x3, 5x5, 3x3 max pooling
      * 降低計算量：直接3x3, 5x5計算量太大, 在這之前先經過1x1到Channel較低的feature map再3x3, 5x5
      * 四條路線的最後H, W都一樣(使用padding), 而C就是四個concat起來, 所以size = (H,W,C1+C2+C3+C4)
    * 1x1的功能：降維(形成bottleneck)並在過程中跨通道學習特徵、降低運算量及參數量
      * 降維(做Cin*Cout的線性映射)
        1. 假設現在input(H,W,C)為10x10x16
        2. 看著每個channel中位置(5,5)的點
        3. 1x1的一個核(1x1x16)相當於對(5,5,1),(5,5,2)...(5,5,16)做**線性映射運算**(對每個點乘以各自的參數後相加,每個參數其實就是1x1核的其中一個切片)
$in\ output\ feature\ map_1:\\new\ (5,5,1)=w_{1,1}(5,5,1)+w_{1,2}(5,5,2)...+w_{1,16}(5,5,16)$
        4. 若output channel為10, 則
$new\ (5,5,2)=w_{2,1}(5,5,1)+w_{2,2}(5,5,2)...+w_{1,16}(2,5,16)\\...\\new\ (5,5,10)=w_{10,1}(5,5,1)+w_{10,2}(5,5,2)...+w_{10,16}(2,5,16)$
        5. 其實就是16 to 10 neurons 的fc (16*10)
        6. 類推到其他所有點(1,1)~(10,10)
      * a channel-wise dense layer that **learn cross channel features**
      * 
      



  * **residual block**
    * skip connections between many layers so that adding more layers won't worsen performance
    * create specail path for the gradient to flow back to ealier layer more easily
    * 通常都用add/concat來連接
    * **應用**
      1. in fully conv networks (連接於對稱的位置)
        * to combine information from deep and shallow layers to produce pixel-wise segmentation maps. 
        * 可以幫助後面的layer還原原圖片物件之位置, 大小等空間資訊 help recover fine spatial information discarded by coarse layers while preserving coarse strucures
      2. in blocks (連接一個block前後)
        * to reuse features from previous layers
  * **attention**

**(2) for NLP**
  1. attention, Transfomer


  
###4. NAS
  * 現階段可能還是輸hand-made, 有論文研究NAS實際上還是比Efficient Model還慢很多就算fp operations比較少, 可能因為NAS容易產生太多network fragmentations

###5. Custom Model
####Conv Model

**(1) How to choose the number of layers and units?**
  * start small
  * gradually increase model size until validation error stop improving

**(2) Deeper or wider?**
  * given constant number of parameters, more layers or more units per layer?
  * **usually deeper than wider**
    * deeper for conv net because bigger receptive filed
    * however, a very tall and skin model is hard to optimize
      * uses **Resnet Block** to skip connections

**(3) Conv kernel size?**
  * **3x3 or 1x1** kernels work the best
    * 3x3 too small? 
      * stack 3x3 on top of each other to reach larger receptive field
        * 5x5: 2 layer of 3x3
        * 7x7: 3 layer of 3x3
    * 1x1 can reduce dimentionality and computation
      * combine 1x1, 3x3

**(4) stride size?**
  * 1 for preserving resolution
  * 2 for downsampling without pooling
  * 1/2 for upsampling
    * transpose convolution/deconvolution

**(5) pooling size**
  * common setting:
    * max pooling
    * 2x2
    * same padding

**(6) activation function**
  * choose ReLU except for output layer

**(7) regularization**
  * use L2 weight decay and dropout between fc
  
**(8) batch size**
  * image recognition: 32
  * noisy gradient: larger
  * stuck in local minimum or out of memory: smaller


**Other Practical Guidelines**
https://www.youtube.com/watch?v=g2vlqhefADk&t=330 較for 硬體
1. giving memory limit, double the number of feature maps and downscale the input by two or vice versa
  * C,H,W => 2C, H/2, W/2 can downsclae memory usage by 2
  * C,H,W => 4C, H/2, W/2 can have constant memory usage throughout entire network and each layer can share the fair size of memory allocation
  * have downside, see paper
2. 3-way seperable FIR-IIR filters for the purpose of line buffer minization


#Note
###1. Model Speed Factor
* floating-point operations 
* memory access
* platform characteristics

###2. 
