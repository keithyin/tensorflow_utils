from __future__ import print_function
import numpy as np
import datetime
import pandas as pd

ONE_DAY = datetime.timedelta(days=1)


def data_generate(date_begin, date_end, per_day_records=1):
    """
    Args:
        date_begin: 20211009
        date_end: 20211031
    Returns: [[event_date, cvr, delayed_days, cvt_date], ]
    """
    # today, cvr, cvt_date
    cur_day = datetime.datetime.strptime(str(date_begin), "%Y%m%d")
    end_day = datetime.datetime.strptime(str(date_end), "%Y%m%d")
    sample = []
    while cur_day <= end_day:
        cur_dates = [cur_day.strftime("%Y%m%d")] * per_day_records
        converts = np.random.binomial(1, p=0.2, size=per_day_records).tolist()
        delayed_days = np.random.exponential(scale=4.0, size=per_day_records).astype(np.int).tolist()
        cvt_date = [(cur_day + d * ONE_DAY).strftime("%Y%m%d") for d in delayed_days]
        sample.extend(list(zip(cur_dates, converts, delayed_days, cvt_date)))
        cur_day += ONE_DAY
    sample = pd.DataFrame(sample, columns=['event_date', 'cvt', 'delayed_days', 'cvt_date'])
    return sample


def train_data_generate(sample, train_date):
    """
    Args:
        sample: data get from data_generate
        train_date: '20211010'
    Returns:
    """
    train_date = str(train_date)
    assert isinstance(sample, pd.DataFrame)
    sample = sample[sample['event_date'] <= train_date].reset_index()
    records = []
    # label, delayed_days
    for _, record in sample.iterrows():
        print(record)
        if record['cvt_date'] > train_date or record['cvt'] == 0:
            label = 0
            delayed_day = (datetime.datetime.strptime(str(train_date), "%Y%m%d") -
                           datetime.datetime.strptime(record['event_date'], "%Y%m%d")).days
            records.append([label, delayed_day])
        else:
            records.append([record['cvt'], record['delayed_days']])
    return pd.DataFrame(records, columns=['label', 'delayed_days'])


if __name__ == '__main__':
    sample = data_generate("20211001", "20211002", per_day_records=5)
    train_data = train_data_generate(sample, "20211002")
    print(sample)
    print(train_data)
