from collections import namedtuple
import warnings
import os
from download import download
from zipfile import ZipFile
import shutil
import pandas as pd
import re

VALID_DATA_FORMATS = ["UIR", "UIRT"]

# Warning and error messages
WARNING_MOVIE_LENS_HEADER = """MovieLens rating dataset has four columns
    (user id, movie id, rating, and timestamp), but more than four column names are provided.
    Will only use the first four column names."""
WARNING_HAVE_SCHEMA_AND_HEADER = """Both schema and header are provided.
    The header argument will be ignored."""
ERROR_MOVIE_LENS_SIZE = (
    "Invalid data size. Option: {100k, 1m, 10m, or 20m}"
)
ERROR_HEADER = "Header error. At least user and movie column names should be provided"

# 100K data genres index to string mapper. For 1m, 10m, and 20m, the genres labels are already in the dataset.
GENRES = (
    "unknown",
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
)

MovieLens = namedtuple("MovieLens", ["url", "unzip", "path", "sep", "item_path", "item_sep", "has_header"])
ML_DATASETS = {
    "100K": MovieLens(
        "https://files.grouplens.org/datasets/movielens/ml-100k.zip",
        False,
        "ml-100k/u.data",
        "\t",
        "ml-100k/u.item",
        "|",
        "False",
    ),
    "1M": MovieLens(
        "http://files.grouplens.org/datasets/movielens/ml-1m.zip",
        True,
        "ml-1m/ratings.dat",
        "::",
        "ml-1m/movies.dat",
        "::",
        "False",
    ),
    "10M": MovieLens(
        "http://files.grouplens.org/datasets/movielens/ml-10m.zip",
        True,
        "ml-10M100K/ratings.dat",
        "::",
        "ml-10M100K/movies.dat",
        "::",
        "False",
    ),
    "20M": MovieLens(
        "http://files.grouplens.org/datasets/movielens/ml-20m.zip",
        True,
        "ml-20m/ratings.csv",
        ",",
        "ml-20m/movies.csv",
        ",",
        "True",
    ),
}


def load_pandas_df(
    size="100k",
    header=("userID", "itemID", "rating", "timestamp"),
    title_col=None,
    genres_col=None,
    year_col=None,
):
    """Loads the MovieLens dataset as pd.DataFrame.

    Download the dataset from https://files.grouplens.org/datasets/movielens, unzip, and load.
    To load movie information only, you can use load_item_df function.

    Args:
        size (str): Size of the data to load. One of ("100k", "1m", "10m", "20m").
        header (list or tuple or None): Rating dataset header.
            If `size` is set to any of 'MOCK_DATA_FORMAT', this parameter is ignored and data is rendered using the 'DEFAULT_HEADER' instead.
        local_cache_path (str): Path (directory or a zip file) to cache the downloaded zip file.
            If None, all the intermediate files will be stored in a temporary directory and removed after use.
            If `size` is set to any of 'MOCK_DATA_FORMAT', this parameter is ignored.
        title_col (str): Movie title column name. If None, the column will not be loaded.
        genres_col (str): Genres column name. Genres are '|' separated string.
            If None, the column will not be loaded.
        year_col (str): Movie release year column name. If None, the column will not be loaded.
            If `size` is set to any of 'MOCK_DATA_FORMAT', this parameter is ignored.

    Returns:
        pandas.DataFrame: Movie rating dataset.


    **Examples**

    .. code-block:: python

        # To load just user-id, item-id, and ratings from MovieLens-1M dataset,
        df = load_pandas_df('1m', ('UserId', 'ItemId', 'Rating'))

        # To load rating's timestamp together,
        df = load_pandas_df('1m', ('UserId', 'ItemId', 'Rating', 'Timestamp'))

        # To load movie's title, genres, and released year info along with the ratings data,
        df = load_pandas_df('1m', ('UserId', 'ItemId', 'Rating', 'Timestamp'),
            title_col='Title',
            genres_col='Genres',
            year_col='Year'
        )
    """
    size = size.upper()
    if size not in ML_DATASETS: 
        raise ValueError(f"Size: {size}. " + ERROR_MOVIE_LENS_SIZE)

    if len(header) < 2:
        raise ValueError(ERROR_HEADER)
    elif len(header) > 4:
        warnings.warn(WARNING_MOVIE_LENS_HEADER)
        header = header[:4]

    movie_col = header[1]

    #Download
    current_path = os.path.abspath(os.getcwd())
    
    filepath = os.path.join(current_path, "data/ml-{}.zip".format(size))
    datapath, item_datapath = download_and_extract(size, filepath)

    # Read item data
    item_df = load_item_df(
            size, item_datapath, movie_col, title_col, genres_col, year_col
        )
    #Read rating data
    df = pd.read_csv(
            datapath,
            sep=ML_DATASETS[size].sep,
            names=header,
            usecols=[*range(len(header))],
            header=0 if ML_DATASETS[size].has_header else None,
        )
    # Convert 'rating' type to float
    if len(header) > 2:
        df[header[2]] = df[header[2]].astype(float)

    # Merge rating df w/ item_df
    if item_df is not None:
        df = df.merge(item_df, on=header[1])

    return df

def load_item_df(size, item_datapath, movie_col, title_col, genres_col, year_col):
    """Loads Movie info"""
    if title_col is None and genres_col is None and year_col is None:
        return None

    item_header = [movie_col]
    usecols = [0]

    # Year is parsed from title
    if title_col is not None or year_col is not None:
        item_header.append("title_year")
        usecols.append(1)

    genres_header_100k = None
    if genres_col is not None:
        # 100k data's movie genres are encoded as a binary array (the last 19 fields)
        # For details, see https://files.grouplens.org/datasets/movielens/ml-100k-README.txt
        if size == "100k":
            genres_header_100k = [*(str(i) for i in range(19))]
            item_header.extend(genres_header_100k)
            usecols.extend([*range(5, 24)])  # genres columns
        else:
            item_header.append(genres_col)
            usecols.append(2)  # genres column
    

    item_df = pd.read_csv(
        item_datapath,
        sep=ML_DATASETS[size].item_sep,
        names=item_header,
        usecols=usecols,
        header=0 if ML_DATASETS[size].has_header else None,
        encoding="ISO-8859-1",

    )

    # Convert 100k data's format: '0|0|1|...' to 'Action|Romance|..."
    if genres_header_100k is not None:
        item_df[genres_col] = item_df[genres_header_100k].values.tolist()
        item_df[genres_col] = item_df[genres_col].map(
            lambda l: "|".join([GENRES[i] for i, v in enumerate(l) if v == 1])
        )

        item_df.drop(genres_header_100k, axis=1, inplace=True)

    # Parse year from movie title. Note, MovieLens title format is "title (year)"
    # Note, there are very few records that are missing the year info.
    if year_col is not None:

        def parse_year(t):
            parsed = re.split("[()]", t)
            if len(parsed) > 2 and parsed[-2].isdecimal():
                return parsed[-2]
            else:
                return None

        item_df[year_col] = item_df["title_year"].map(parse_year)
        if title_col is None:
            item_df.drop("title_year", axis=1, inplace=True)

    if title_col is not None:
        item_df.rename(columns={"title_year": title_col}, inplace=True)

    return item_df

def download_and_extract(size, dest_path):
    """Downloads and extracts MovieLens rating and item datafiles if they don’t already exist"""
    dirs, _ = os.path.split(dest_path)
    if not os.path.exists(dirs):
        os.makedirs(dirs)

    rating_filename = ML_DATASETS[size].path
    rating_path = os.path.join(dirs, rating_filename)
    item_filename = ML_DATASETS[size].item_path
    item_path = os.path.join(dirs, item_filename)

    if not os.path.exists(rating_path) or not os.path.exists(item_path):
        download_movielens(size, dest_path)
        extract_movielens(size, rating_path, item_path, dest_path)

    return rating_path, item_path

def download_movielens(size, dest_path):
    """Downloads MovieLens datafile.

    Args:
        size (str): Size of the data to load. One of ("100k", "1m", "10m", "20m").
        dest_path (str): File path for the downloaded file
    """
    if size not in ML_DATASETS:
        raise ValueError(f"Size: {size}. " + ERROR_MOVIE_LENS_SIZE)

    url = ML_DATASETS[size].url
    dirs, file = os.path.split(dest_path)
    download(url, file, work_directory=dirs)


def extract_movielens(size, rating_path, item_path, zip_path):
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
    os.makedirs('ml-{}'.format(size.lower()), exist_ok=True)  # Create the 'ml-1m' folder, if it doesn't already exist
    print("ZIPPATH= ", zip_path)

    with ZipFile(zip_path, "r") as z:
        print(1)
        with z.open(ML_DATASETS[size].path) as zf, open(rating_path, "wb") as f:
            
            shutil.copyfileobj(zf, f)
        with z.open(ML_DATASETS[size].item_path) as zf, open(item_path, "wb") as f:
            shutil.copyfileobj(zf, f)


    os.remove(zip_path)

