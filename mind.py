from collections import namedtuple
import warnings
import os
from download import download
from zipfile import ZipFile
import shutil
import pandas as pd
import random

ERROR_MIND_SIZE = (
    "Invalid data size. Option: {small, large, demo}"
)


Mind = namedtuple("Mind", ["train_url", "dev_url", "behaviors_train_path", "news_train_path", "behaviors_dev_path", "news_dev_path"])
MIND_DATASETS = {
    "small": Mind(
        "https://recodatasets.z20.web.core.windows.net/newsrec/MINDsmall_train.zip",
        "https://recodatasets.z20.web.core.windows.net/newsrec/MINDsmall_dev.zip",
        "MINDsmall_train/behaviors.tsv",
        "MINDsmall_train/news.tsv",
        "MINDsmall_dev/behaviors.tsv",
        "MINDsmall_dev/news.tsv",
    ),
    "large": Mind(
        "https://recodatasets.z20.web.core.windows.net/newsrec/MINDlarge_train.zip",
        "https://recodatasets.z20.web.core.windows.net/newsrec/MINDlarge_dev.zip",
        "MINDlarge_train/behaviors.tsv",
        "MINDlarge_train/news.tsv",
        "MINDlarge_dev/behaviors.tsv",
        "MINDlarge_dev/news.tsv",
    ),
    "demo": Mind(
        "https://recodatasets.z20.web.core.windows.net/newsrec/MINDdemo_train.zip",
        "https://recodatasets.z20.web.core.windows.net/newsrec/MINDdemo_dev.zip",
        "MINDdemo_train/behaviors.tsv",
        "MINDdemo_train/news.tsv",
        "MINDdemo_dev/behaviors.tsv",
        "MINDdemo_dev/news.tsv",
    )
}

def load_pandas_df(
    size="small",
    behaviors_header=None,
    news_header=None,
    npratio=4
):
    size = size.lower()
    #Download dataset if not already downloaded
    current_path = os.path.abspath(os.getcwd())
    
    train_path = os.path.join(current_path, "data/MIND{}_train.zip".format(size))
    dev_path = os.path.join(current_path, "data/MIND{}_dev.zip".format(size))
    behaviors_train_path, behaviors_dev_path, news_train_path, news_dev_path = download_and_extract(size, train_path, dev_path)

    #Load behaviors
    behaviors_train_df = read_behaviors(behaviors_train_path, header=behaviors_header, npratio=npratio) if behaviors_header is not None else read_behaviors(behaviors_train_path, npratio=npratio)
    behaviors_dev_df = read_behaviors(behaviors_dev_path, header=behaviors_header, npratio=npratio) if behaviors_header is not None else read_behaviors(behaviors_dev_path, npratio=npratio)

    #Load news
    news_train_df = read_news(news_train_path, header=news_header) if news_header is not None else read_news(news_train_path)
    news_dev_df = read_news(news_dev_path, header=news_header) if news_header is not None else read_news(news_dev_path)
    
    return behaviors_train_df, behaviors_dev_df, news_train_df, news_dev_df


def read_behaviors(
    behaviors_path,
    header=('ID', 'userID', 'timestamp', 'history', 'impression'),
    npratio=4
):
    df = pd.read_csv(
        behaviors_path, 
        sep='\t', 
        names=header,
        usecols=[*range(len(header))],
    )
    

    df[['positive', 'negative']] = df['impression'].apply(
        lambda x: pd.Series([
            [imp.split('-')[0] for imp in x.split() if imp.split('-')[1] == '1'],  # Positive impressions
            [imp.split('-')[0] for imp in x.split() if imp.split('-')[1] == '0']   # Negative impressions
        ])
    )

    expanded_rows = []
    for _, row in df.iterrows():
        positives = row['positive']
        negatives = row['negative']
        for pos_id in positives:
            sampled_negatives = ' '.join(get_sample(negatives, npratio))  
            expanded_rows.append({
                header[0]: row[header[0]],
                header[1]: row[header[1]],
                header[2]: row[header[2]],
                header[3]: row[header[3]],
                'positive': pos_id,
                'negative': sampled_negatives
            })

    # Convert to DataFrame
    expanded_df = pd.DataFrame(expanded_rows)
        
    return expanded_df

def read_news(
    news_path,
    header=('newsID', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities')
):
    df = pd.read_csv(
        news_path, 
        sep='\t', 
        names=header,
        usecols=[*range(len(header))]
    )

    return df

def get_sample(all_elements, num_sample):
    if num_sample > len(all_elements):
        return random.sample(all_elements * (num_sample // len(all_elements) + 1), num_sample)
    else:
        return random.sample(all_elements, num_sample)



def download_and_extract(size, train_path, dev_path):
    """Downloads and extracts Mind if they don’t already exist"""
    dirs, _ = os.path.split(train_path)
    if not os.path.exists(dirs):
        os.makedirs(dirs)

    behaviors_train_name = MIND_DATASETS[size].behaviors_train_path
    behaviors_dev_name = MIND_DATASETS[size].behaviors_dev_path
    news_train_name = MIND_DATASETS[size].news_train_path
    news_dev_name = MIND_DATASETS[size].news_dev_path

    behaviors_train_path = os.path.join(dirs, behaviors_train_name)
    behaviors_dev_path = os.path.join(dirs, behaviors_dev_name)
    news_train_path = os.path.join(dirs, news_train_name)
    news_dev_path = os.path.join(dirs, news_dev_name)



    if not os.path.exists(behaviors_train_path) or not os.path.exists(behaviors_dev_path) or not os.path.exists(news_train_path) or not os.path.exists(news_dev_path):
        download_mind(size, train_path, dev_path)
        print("{}\n{}\n{}\n{}\n".format(behaviors_train_path, behaviors_dev_path, news_train_path, news_dev_path))

        extract_mind(size, behaviors_train_path, behaviors_dev_path, news_train_path, news_dev_path, train_path, dev_path)

    return behaviors_train_path, behaviors_dev_path, news_train_path, news_dev_path

def download_mind(size, train_path, dev_path):
    """Downloads Mind.

    Args:
        size (str): Size of the data to load. One of ("large", "small", "demo").
        dest_path (str): File path for the downloaded file
    """
    if size not in MIND_DATASETS:
        raise ValueError(f"Size: {size}. " + ERROR_MIND_SIZE)

    train_url = MIND_DATASETS[size].train_url
    dev_url = MIND_DATASETS[size].dev_url

    dirs, _ = os.path.split(train_path)
    download(train_url, train_path, work_directory=dirs)
    download(dev_url, dev_path, work_directory=dirs)



def extract_mind(size, behaviors_train_path, behaviors_dev_path, news_train_path, news_dev_path, train_zip_path, dev_zip_path):
    """Extract MovieLens rating and item datafiles from the MovieLens raw zip file.

    To extract all files instead of just rating and item datafiles,
    use ZipFile's extractall(path) instead.

    Args:
        size (str): Size of the data to load. One of ("100k", "1m", "10m", "20m").
        rating_path (str): Destination path for rating datafile
        item_path (str): Destination path for item datafile
        zip_path (str): zipfile path
    """
    os.chdir('data')  # Change to the data folder
    os.makedirs('MIND{}_train'.format(size.lower()), exist_ok=True) 
    os.makedirs('MIND{}_dev'.format(size.lower()), exist_ok=True) 
    
    
    with ZipFile(train_zip_path, "r") as z:
        with z.open('behaviors.tsv') as zf, open(behaviors_train_path, "wb") as f:
            shutil.copyfileobj(zf, f)
        with z.open('news.tsv') as zf, open(news_train_path, "wb") as f:
            shutil.copyfileobj(zf, f)

    with ZipFile(dev_zip_path, "r") as z:
        with z.open('behaviors.tsv') as zf, open(behaviors_dev_path, "wb") as f:
            
            shutil.copyfileobj(zf, f)
        with z.open('news.tsv') as zf, open(news_dev_path, "wb") as f:
            shutil.copyfileobj(zf, f)


    os.remove(train_zip_path)
    os.remove(dev_zip_path)


if __name__ == "__main__":
    size = "small"
    behaviors_train_df, behaviors_dev_df, news_train_df, news_dev_df = load_pandas_df(size)
    print(behaviors_train_df)
    print(news_train_df)

    