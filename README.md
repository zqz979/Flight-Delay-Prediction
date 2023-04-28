# Flight-Delay-Prediction

## Instructions for Use
### For all options these steps will need to be taken
* Unzip data.zip
* Dependencies will need to be installed. For example, yfinance is likely not yet installed
* Change the path of files such as data.csv to match your file system
### Option 1 (recommended)
* Fork the repository from https://github.com/zqz979/Flight-Delay-Prediction
### Option 2
* Upload utils.py to the Colab file system
* Upload data to the Colab file system
### Option 3
* Copy paste everything from utils.py into a notebook
* Change syntax to avoid errors (i.e., remove "import utils", and "utils.*")
## Notes
* To reproduce results, utils.load_data() will need to be called with the associated parameters for some notebooks. Examples are given in "KNN Model.ipynb"
* For convenience, here are the tests:
    * utils.load_data(resample=True,subsample=1.0,bts_only=False,covid=None,stock_only=False,select_features=False)
    * utils.load_data(resample=True,subsample=0.25,bts_only=False,covid=None,stock_only=False,select_features=False)
    * utils.load_data(resample=True,subsample=1.0,bts_only=True,covid=None,stock_only=False,select_features=False)
    * utils.load_data(resample=True,subsample=1.0,bts_only=True,covid=False,stock_only=False,select_features=False)
    * utils.load_data(resample=True,subsample=1.0,bts_only=True,covid=True,stock_only=False,select_features=False)
    * utils.load_data(resample=True,subsample=1.0,bts_only=False,covid=None,stock_only=True,select_features=False)
    * utils.load_data(resample=False,subsample=1.0,bts_only=False,covid=None,stock_only=False,select_features=False)
    * utils.load_data(resample=False,subsample=1.0,bts_only=False,covid=None,stock_only=False,select_features=True)