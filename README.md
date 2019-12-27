dddddd*EXAMPLE*
* Model: Transformer, fairseq
* Data: ASPEC

# preprocessing
```
$ cd fairseq
$ TEXT=ASPEC/data/ # 前処理したいデータ
$ python preprocess.py \
    --source-lang ja \ 
    --target-lang en \
    --trainpref $TEXT/train-1 \
    --validpref $TEXT/dev \
    --testpref $TEXT/test \
    --destdir data-bin/aspec.ja-en \ # 前処理したデータの保存先
    --thresholdtgt 10 \ # 目的言語側で出現頻度が10以下の単語を<unk>に置き換え
    --thresholdsrc 10 # 原言語側で出現頻度が10以下の単語を<unk>に置き換え
```

# training
```
$ mkdir -p checkpoints/trans
$ python train.py data-bin/aspec.ja-en \ # preprocessingで処理したデータの保存先
    --lr 0.1 \
    --clip-norm 0.1 \
    --dropout 0.2 \
    --arch transformer \
    --encoder-embed-dim 300 \
    --decoder-embed-dim 300 \
    --encoder-layers 4 \
    --decoder-layers 4 \
    --encoder-attention-heads 5 \
    --decoder-attention-heads 5 \
    --save-dir checkpoints/trans \  # 学習結果の保存先
    --max-sentences 64 \
    --max-epoch 50
```

# test (use the best model)
```
$ python generate.py data-bin/aspec.ja-en \ # preprocessingで処理したデータの保存先
    --path checkpoints/trans/checkpoint_best.pt \　# 学習結果の保存先から、bestなmodelを呼び出す
    --batch-size 128 \
    --beam 5
```

・view at datatable
```
# please set your text generated from "generate.py" in "output2csv.py"
# do below command, you can get csv file
$ python output2csv.py FAIRSEQ_OUTPUTS
```
