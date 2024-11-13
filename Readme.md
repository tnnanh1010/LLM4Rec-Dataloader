!mkdir data
!cd data

# Download MindSmall
!wget https://recodatasets.z20.web.core.windows.net/newsrec/MINDsmall_train.zip
!wget https://recodatasets.z20.web.core.windows.net/newsrec/MINDsmall_dev.zip
!unzip MINDsmall_train.zip -d MINDsmall_train
!unzip MINDsmall_dev.zip -d MINDsmall_dev

!rm MINDsmall_train.zip
!rm MINDsmall_dev.zip

!echo 'Data download finish.'

# Download MindLarge
!wget https://recodatasets.z20.web.core.windows.net/newsrec/MINDlarge_train.zip
!wget https://recodatasets.z20.web.core.windows.net/newsrec/MINDlarge_dev.zip
!wget https://recodatasets.z20.web.core.windows.net/newsrec/MINDlarge_test.zip

!unzip MINDlarge_train.zip -d MINDlarge_train
!unzip MINDlarge_dev.zip -d MINDlarge_dev
!unzip MINDlarge_test.zip -d MINDlarge_test

!rm MINDllarge_train.zip
!rm MINDlarge_dev.zip
!rm MINDlarge_test.zip

!echo 'Data download finish.'

# Download MoviesLens

!wget https://files.grouplens.org/datasets/movielens/ml-1m.zip

!unzip ml-1m.zip
!rm ml-1m.zip


!wget https://files.grouplens.org/datasets/movielens/ml-10m.zip

!unzip ml-10m.zip
!rm ml-10m.zip


!wget https://files.grouplens.org/datasets/movielens/ml-100k.zip

!unzip ml-100k.zip
!rm ml-100k.zip

https://files.grouplens.org/datasets/movielens/ml-20m.zip