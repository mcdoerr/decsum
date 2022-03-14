# Decision-Focused Summarization

Replication for CSE 517 of Decision-Focused Summarization [paper link](https://arxiv.org/abs/2109.06896). 

Note that running this code may be very slow. Our trained models were too large to upload here, please reach out to mcdoerr@uw.edu for the trained models.

## Instructions for Running
Create env with conda:
```
conda create -n yelp python=3.7.6
```
Then install packages with:

```
cat requirements.txt | sed -e '/^\s*#.*$/d' -e '/^\s*$/d' | xargs -n 1 python -m pip install
# download spacy package
python -m spacy download en_core_web_sm

# If you are using RTX3090, try the following step to install pytorch
pip install torch==1.7.0+cu110 -f https://download.pytorch.org/whl/torch_stable.html
```

## Data Preprocessing
We use an updated dataset from https://www.yelp.com/dataset/download as compared to the original paper. Uncompress it to `YELP_DATA_DIR`. The size should be about ~12 GB 
```
python -m preprocess.yelp_preprocess [--yelp_data_dir YELP_DATA_DIR] [--output_dir OUTPUT_DIR]
```

## Train Longformer model
>It takes about an hours to train longformer on GTX 1080 (11GB) with half precision, with sequence length 100. Sequence length is updated from the original model to fit into memory.
```
bash scripts/train_transformer.sh
```
You can check training log here `${OUTPUT_DIR}/logs/` with `tensorboard`.
Trained model will be saved to path like this `${OUTPUT_DIR}/version_27-12-2021--16-59-15/checkpoints/epoch=1-val_loss=0.12.ckpt`. These can be loaded back for evaluation purposes with "load_model_from_ckpt" function appended to the end of the longformer model.

## Run DecSum
> This step takes about 145 hours on GTX 1080Ti with 11GB.

```
# at base Directory
bash scripts/sentence_selection.sh
```
The DecSum summaries will be saved at `${RES_DIR}/models/sentence_select/selected_sentence/yelp/50reviews/test/Transformer/window_1_DecSum_WD_sentbert_50trunc_1_1_1/best/1/text_.csv`.

*_MSE with True Label_* metric will be store at `${RES_DIR}/models/sentence_select/results/yelp/50reviews/test/Transformer/window_1_DecSum_WD_sentbert_50trunc_1_1_1/best/1/text_.csv`.

# Citation For Original Paper
```
@inproceedings{hsu-tan-2021-decision,
    title = "Decision-Focused Summarization",
    author = "Hsu, Chao-Chun  and
      Tan, Chenhao",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.10",
    doi = "10.18653/v1/2021.emnlp-main.10",
    pages = "117--132",
}
```