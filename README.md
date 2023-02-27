# Flight-Delay-Prediction

Proposal Link:
https://pennstateoffice365-my.sharepoint.com/:w:/g/personal/bie5065_psu_edu/Ea7yL5wxb6xPi3DjimbtIuIBd5bvh_hbcatg4YRV-kIx7w?e=4%3AbT5Cic&at=9&CID=dd61a598-d9dc-e8b1-e691-55d7eff8360f

## Notes
./data/bts_data.csv has all the bts data with all the features I think we should look at. Features that were not included include the ones that had many rows without values, ones that seemed repetitive like departure time vs departure time block, and ones that seem like cheating like flight arrival time.

./data/stock_data.csv has all the stock info

./data/all_data.csv is the stocks with the bts data. Something to note is that I think no weekends are there since there is no stock data for weekends and stuff like that.

For preprocessing note that there is a max_unqiues which gets rid of big categorical features. I was thinking maybe label encoding will work for k best because the feature values might not matter, just how important the whole feature is. Also not really sure how k best would work with one hot because 1 feature becomes 20 features. After k best maybe we do one hot on remaining features. Another solution is to randomly sample like 10% of data and use that since I think there are still over 1 mil points. Best solution might be to consider most common categories and the other categories in a feature get grouped together as a "rare" category for one hot.

Example usage in ann.py