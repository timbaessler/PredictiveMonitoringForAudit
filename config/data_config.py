import os
from sys import platform

if platform == "linux":
    path = '/content/drive/MyDrive/PrPM4CA/notebooks/three-lines-predictive-process-mining'
elif platform == "win32":
    path = r'G:\Meine Ablage\PrPM4CA\notebooks\three-lines-predictive-process-mining'
else:
    raise RuntimeError

bpic_2018_dict = dict({'bpi_path' : os.path.join(path, 'data', 'BPIC2018'),
                       'processed_path': os.path.join(path, 'data', 'BPIC2018', 'processed'),
                       'predict_path': os.path.join(path, 'data', 'BPIC2018', 'predictions'),
                       'labelled_log_path': os.path.join(path, 'data', 'BPIC2018', 'processed',
                                                         'bpic2018_labelled.feather'),
                       'res_path': os.path.join(path, 'data', 'BPIC2018', 'results'),
                       'static_cat_cols': list(['case:young farmer','case:penalty_AJLP','case:small farmer',
                                                'case:penalty_BGP', 'case:department', 'case:penalty_C16',
                                                'case:penalty_BGK', 'case:penalty_AVUVP', 'case:penalty_CC',
                                                'case:penalty_AVJLP', 'case:penalty_C9', 'case:cross_compliance',
                                                'case:rejected', 'case:penalty_C4', 'case:penalty_AVGP',
                                                'case:penalty_ABP', 'case:penalty_B6', 'case:penalty_B4',
                                                'case:penalty_B5', 'case:penalty_AVBP', 'case:penalty_B2',
                                                'case:selected_risk', 'case:penalty_B3', 'case:selected_manually',
                                                'case:penalty_AGP', 'case:penalty_B16', 'case:penalty_GP1',
                                                'case:basic payment', 'case:penalty_B5F', 'case:penalty_V5',
                                                'case:payment_actual0', 'case:redistribution', 'case:penalty_JLP6',
                                                'case:penalty_JLP7', 'case:penalty_JLP5', 'case:penalty_JLP2',
                                                'case:penalty_JLP3', 'case:penalty_JLP1']),
                       'dynamic_cat_cols': list(['org:resource', 'concept:name', 'success', 'doctype', 'subprocess',
                                                 'note',]),
                       'num_cols': list(['case:penalty_amount0', 'month', 'weekday', 'hour', 'time_since_first_event',
                                         'time_since_last_event', 'case:amount_applied0', 'case:number_parcels',
                                         'case:area'])})

bpic_2019_dict = dict({'bpi_path' : os.path.join(path, 'data', 'BPIC2019'),
                       'processed_path': os.path.join(path, 'data', 'BPIC2019', 'processed'),
                       'predict_path': os.path.join(path, 'data', 'BPIC2019', 'predictions'),
                       'labelled_log_path': os.path.join(path, 'data', 'BPIC2019', 'processed',
                                                         'bpic_2019_labelled.feather'),
                       'res_path': os.path.join(path, 'data', 'BPIC2019', 'results'),
                       'static_cat_cols': list(['case:Spend area text', 'case:Company',
                                                'case:Document Type', 'case:Sub spend area text',
                                                #'case:Vendor',
                                                'case:Item Type', 'case:Item Category',
                                                'case:Spend classification text',
                                                # 'case:Name',
                                                'case:GR-Based Inv. Verif.', 'case:Goods Receipt']),
                       'dynamic_cat_cols': list(['org:resource', 'concept:name']),
                       'num_cols': list(['Cumulative net worth (EUR)', 'year', 'month', 'weekday', 'hour',
                                        'time_since_first_event',
                                        'time_since_last_event'])})