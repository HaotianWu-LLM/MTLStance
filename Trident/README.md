# Trident-PLUS: A Improved Version for Trident
This is the improved version for Trident, which infuse both google knowledge and Wikipedia Knowledge. We propose to utilize the background knowledge from Wikipedia and Google about the target to improve zero-shot and few-shot stance detection.
Part of this code is referenced at [here](https://github.com/zihaohe123/wiki-enhanced-stance-detection/tree/main)
## Dataset Preparation
1. In this word, we experiment on one ZSSD/FSSD dataset- <em>VAST<em>, which can be is publicly available at [here](https://github.com/emilyallaway/zero-shot-stance/tree/master/data/VAST)
2. "wiki_dict.pkl" and "google_dict.pkl" are two offline documents that we use to retrieve background knowledge towards "Targets", if you want to use it in your own dataset, you need to update these two document with the "Target" in your dataset.
## Installation
```
pip install -r requirements.txt
```
## Some Kindly Suggestions
1. Multi-task Learning is usually difficult to converge to an optimal point. Therefore, suitable weights for different task are important. The "Target Prediction" is a difficult task (Because there are a large number of classes), thus we suggest to set its loss weight to lower than <em>1e-3<em>.
2. [here](https://github.com/median-research-group/LibMTL) is a easy-to-use library, which contain many loss-balance and gradient-balance method, which may help to improve the MTL performence. CAGrad often works better in my past MTL experience. 
3. Larger batch_size is useful, but it may consume large GPU Video Memory.
4. You can freeze first half of BERT's layer, the performance will not greatly decrease. But not freezing all layers works best.
5. What's more, the closer the training corpus of the pre-trained model chosen is to your actual data, the better.
6. Prompt Learning dose not work well in this MTL. The Prompt we used is "The Stance of (t) towards (d) with knowledge (w+g) is ". g is the knowledge of google.
## Run
VAST, zero/few-shot stance detection
```angular2html
python run_vast.py
```
## Some Important Ablation Studies (Sort by priority)
(1) The selection of the number of experts (from 1 to 4) \
(2) Remove the google branch, wiki branch and tweet branch, respectively.
(3) Set the weight of "loss_target", "loss_individual", "loss_margin" to 0, respectively.
(4) Important Note: Set the patience to 2 is enough.
