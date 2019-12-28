# My setting in paper
* Model: Transformer, fairseq
* Data: ASPEC-JE

# preprocessing
```
$ cd fairseq
$ TEXT=ASPEC/data/ # path of data
$ python preprocess.py \
    --source-lang ja \ 
    --target-lang en \
    --trainpref $TEXT/train-1 \
    --validpref $TEXT/dev \
    --testpref $TEXT/test \
    --destdir data-bin/aspec.ja-en \ # save path
    --thresholdtgt 10 \ # min-count to replace <unk> in target(output) language
    --thresholdsrc 10 # min-count to replace <unk> in source(input) language
```

# training
```
$ mkdir -p checkpoints/trans
$ python train.py data-bin/aspec.ja-en \ # path of the data preprocessed
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
    --save-dir checkpoints/trans \  # save path of training model
    --max-sentences 64 \
    --max-epoch 50
```

# test (use the best model)
```
$ python generate.py data-bin/aspec.ja-en \ # path of the data preprocessed
    --path checkpoints/trans/checkpoint_best.pt \　# use best model
    --batch-size 128 \
    --beam 5
```

# Monte Carlo Dropout Sampling
I use Monte Carlo Dropout (MC Dropout) sampling.
This method uses dropout in **test step** to obtain **model uncertainty**.
I think that model uncertainty is useful for **confidence estimation**.
If you use MC Dropout, you have to do some settings.
## Use 'TargetSearch' class [[link](https://github.com/a1da4/conf_nmt/blob/master/fairseq/fairseq/sequence_generator.py#L100)]
```
#conf_nmt/fairseq/fairseq/sequence_generator.py

self.search = search.TargetSearch(tgt_dict)
#self.search = search.BeamSearch(tgt_dict) # default
```

## Use Dropout in test step (transformer settings) [[link](https://github.com/a1da4/conf_nmt/blob/master/fairseq/fairseq/models/transformer.py#L215)]
```
#conf_nmt/fairseq/fairseq/models/transformer.py

#x = F.dropout(x, p=self.dropout, training=self.training) # default
x = F.dropout(x, p=0.2) # use dropout everytime
```

・view at datatable
```
# please set your text generated from "generate.py" in "output2csv.py"
# do below command, you can get csv file
$ python output2csv.py FAIRSEQ_OUTPUTS
```
