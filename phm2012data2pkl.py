import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# 更换为自己的数据路径
source_path = Path(r'/home/knd/base_rul/phm_ieee_2012_data_challenge_dataset')
assert source_path.exists(), f"错误：文件夹 {source_path} 不存在！"

RUL_dict = {'Bearing1_1': 0, 'Bearing1_2': 0,
            'Bearing2_1': 0, 'Bearing2_2': 0,
            'Bearing3_1': 0, 'Bearing3_2': 0,
            'Bearing1_3': 573, 'Bearing1_4': 33.9, 'Bearing1_5': 161, 'Bearing1_6': 146, 'Bearing1_7': 757,
            'Bearing2_3': 753, 'Bearing2_4': 139, 'Bearing2_5': 309, 'Bearing2_6': 129, 'Bearing2_7': 58,
            'Bearing3_3': 82}

phm_dataset = {bearing_name: {} for bearing_name in RUL_dict.keys()}

# ['Learning_set', 'Full_Test_Set'] or ['Learning_set', 'Test_set']
for path in ['Learning_set', 'Full_Test_Set']:
    assert (source_path / path).exists(), f"错误：文件夹 {source_path} 不存在！"

    bearings_names = [f.name for f in (source_path / path).iterdir()]
    bearings_names.sort()
    for bearings_name in bearings_names:
        print(f'{bearings_name}加载中！')
        file_names = [f.name for f in (source_path / path / bearings_name).iterdir()]
        file_names.sort()
        bearing_data = None
        for file_name in file_names:
            if 'acc' in file_name:
                file_path = source_path / path / bearings_name / file_name
                sep = ';' if bearings_name == 'Bearing1_4' else ','
                df = pd.read_csv(file_path, sep=sep, header=None)
                data = np.array(df.loc[:, 4:6])
                data = data[np.newaxis, :, :]
                if bearing_data is None:
                    bearing_data = data
                else:
                    bearing_data = np.append(bearing_data, data, axis=0)
        phm_dataset[bearings_name]['RUL'] = RUL_dict[bearings_name]
        phm_dataset[bearings_name]['quantity'] = bearing_data.shape[0]
        phm_dataset[bearings_name]['data'] = bearing_data

        print(f'{bearings_name}加载完毕！')

folder_path = Path(r"./pkl_data")  # 创建 Path 对象
folder_path.mkdir(parents=True, exist_ok=True)  # 自动创建文件夹（如果已存在则不会报错）

with open(folder_path / 'phm_dataset.pkl', 'wb') as f:
    pickle.dump(phm_dataset, f)
