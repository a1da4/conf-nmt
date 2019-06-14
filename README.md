use fairseq(transformer)

・preprosessing

```
TEXT=ASPEC/data/
python preprocess.py \
    --source-lang ja \ 
    --target-lang en \
    --trainpref $TEXT/train-1 \
    --validpref $TEXT/dev \
    --testpref $TEXT/test \
    --destdir data-bin/aspec.ja-en \
    --thresholdtgt 10 \ # 目的言語側で出現頻度が10以下の単語を<unk>に置き換え
    --thresholdsrc 10 # 原言語側で出現頻度が10以下の単語を<unk>に置き換え
```

・training
$ mkdir -p checkpoints/trans
$ python train.py data-bin/aspec.ja-en \
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
    --save-dir checkpoints/trans \
    --max_sentence 64
    
・test (use the best model)
$ python generate.py data-bin/aspec.ja-en \
    --path checkpoints/trans/checkpoint_best.pt \
    --batch-size 128 \
    --beam 5

reference
https://qiita.com/tkmaroon/items/f60ad171911409eed2af
