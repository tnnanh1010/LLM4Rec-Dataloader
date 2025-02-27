from dataclasses import dataclass
from typing import Optional

@dataclass
class NamlArgs:
    nGPU: int = 1
    seed: int = 0
    prepare: bool = True
    mode: str = "train"
    train_data_dir: str = "/home/vinmike/Documents/GitHub/LLM4Rec-Dataloader/data/MINDsmall_train"
    test_data_dir: str = "/home/vinmike/Documents/GitHub/LLM4Rec-Dataloader/data/MINDsmall_dev"
    custom_abstract_dir: str = ""
    model_dir: str = '/content/model'
    batch_size: int = 32
    npratio: int = 4
    enable_gpu: bool = True
    filter_num: int = 3
    log_steps: int = 100
    epochs: int = 5
    lr: float = 0.0003
    num_words_title: int = 20
    num_words_abstract: int = 50
    user_log_length: int = 50
    word_embedding_dim: int = 300
    glove_embedding_path: str = '/home/vinmike/Documents/GitHub/LLM4Rec-Dataloader/data/glove.840B.300d.txt'
    freeze_embedding: bool = False
    news_dim: int = 400
    news_query_vector_dim: int = 200
    user_query_vector_dim: int = 200
    num_attention_heads: int = 20
    user_log_mask: bool = False
    drop_rate: float = 0.2
    save_steps: int = 10000
    start_epoch: int = 0
    load_ckpt_name: Optional[str] = None
    use_category: bool = True
    use_subcategory: bool = True
    use_abstract: bool = True
    use_custom_abstract: bool = False
    category_emb_dim: int = 100

def parse_naml_args():
  return NamlArgs()

from typing import Optional
from dataclasses import dataclass

@dataclass
class MinerArgs:
    tensorboard_path: str = 'run'
    # Common args
    nGPU: int = 1
    mode: str = 'train'
    model_name: str = "Miner"
    pretrained_tokenizer: str = "vinai/phobert-base"
    user2id_path: str = "/home/vinmike/Documents/GitHub/LLM4Rec-Dataloader/data/MINDsmall_train/user2id.json"
    category2id_path: str = "/home/vinmike/Documents/GitHub/LLM4Rec-Dataloader/data/MINDsmall_train/category2id.json"
    max_title_length: int = 32
    max_sapo_length: int = 64
    his_length: int = 50
    seed: int = 36
    metrics: str = "auc"

    # Data args
    train_path = "/home/vinmike/Documents/GitHub/LLM4Rec-Dataloader/data/MINDsmall_train"
    data_name: str = "News_Recommend_Data"
    train_behaviors_path: str = "/home/vinmike/Documents/GitHub/LLM4Rec-Dataloader/data/MINDsmall_train/behaviors.tsv"
    train_news_path: str = "/home/vinmike/Documents/GitHub/LLM4Rec-Dataloader/data/MINDsmall_train/news.tsv"
    eval_behaviors_path: str = "/home/vinmike/Documents/GitHub/LLM4Rec-Dataloader/data/MINDsmall_dev/behaviors.tsv"
    eval_news_path: str = "/home/vinmike/Documents/GitHub/LLM4Rec-Dataloader/data/MINDsmall_dev/news.tsv"
    category_embed_path: str = None
    
    # Model args
    pretrained_embedding: str = "vinai/phobert-base"
    apply_reduce_dim: bool = True
    use_sapo: bool = True
    word_embed_dim: int = 256
    category_embed_dim: int = 100
    combine_type: str = "linear"
    num_context_codes: int = 32
    context_code_dim: int = 200
    score_type: str = "weighted"
    dropout: float = 0.2
    warmup_steps: str = None

    # Train args
    npratio: int = 4
    train_batch_size: int = 8
    eval_batch_size: int = 64
    dataloader_drop_last: bool = True
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    gradient_accumulation_steps: int = 16
    num_train_epochs: int = 5
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    logging_steps: int = 200
    evaluation_info: str = "metrics"
    eval_steps: int = 400
    fast_eval: str = False
    max_steps: str = None
    freeze_transformer: str = True
    use_sapo: str = True
    lstm_num_layers: int = 1
    lstm_dropout: float = 0.2
    use_category_bias: str = True
    
def parse_miner_args():
    return MinerArgs()