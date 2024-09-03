import pandas as pd
import numpy as np
from .signal_filtering import process_signal 
def minmax(df_temp):
    """
    將資料標準化為 0~1 之間
    """
    return (df_temp - df_temp.min()) / (df_temp.max() - df_temp.min())
def get_data(path, data_type, process=True):
    """
    針對原始數據進行前處理
    :param path: 數據文件路徑
    :param data_type: 數據類型，可以是 'Total', 'Oxy', 或 'Deoxy'
    :param process: 是否進行信號處理
    """
    # 引入資料+去除不必要的欄位
    df = pd.read_csv(path, skiprows=40)
    
    # 根據數據類型選擇要刪除的列
    columns_to_drop = ["BodyMovement", "RemovalMark", 'PreScan', 'Mark']
    if data_type == 'Total':
        columns_to_drop.append('Probe1(Total)')
    elif data_type == 'Oxy':
        columns_to_drop.append('Probe1(Oxy)')
    elif data_type == 'Deoxy':
        columns_to_drop.append('Probe1(Deoxy)')
    else:
        raise ValueError("Invalid data_type. Must be 'Total', 'Oxy', or 'Deoxy'.")
    
    df_drop = df.drop(columns=columns_to_drop)
    df_drop = df_drop.set_index("Time")
    index_list = [int(x.split(':')[1])*60 + float(x.split(':')[2]) for x in df_drop.index]
    df_drop.index = index_list
    try:
        df_drop = df_drop.drop([-999], axis=0)
    except:
        pass
    if process:
        result = pd.DataFrame(process_signal(df_drop))
    result = df_drop.groupby(np.arange(len(df_drop)) // 10).mean()
    result = result.transpose()
    # 針對特例進行的處理
    if result.shape == (99, 126):
        result = result.dropna(axis=0)
    return minmax(result)