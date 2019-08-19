from __future__ import print_function
import pickle
import os.path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import numpy as np




class GDrive:
    def __init__(self, model_name, dataset_name, transfer, data_augmentation):
        self.SAMPLE_SPREADSHEET_ID = '1GECoXg8hU2V1VIvbGdqcZDzcsi16zynRq56L72F6xkM'
        self.SAMPLE_RANGE_NAME = 'auto-ablation'

        self.service = self.get_service()
        self.row, self.col = self.get_coors(model_name, dataset_name, transfer, data_augmentation)


    def get_service(self):
        creds = None
        # The file ./utils/token.pickle stores the user's access and refresh tokens, and is
        # created automatically when the authorization flow completes for the first
        # time.
        SCOPES = ['https://www.googleapis.com/auth/drive', 'https://www.googleapis.com/auth/spreadsheets']

        if os.path.exists('./utils/token.pickle'):
            with open('./utils/token.pickle', 'rb') as token:
                creds = pickle.load(token)
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    './utils/credentials.json', SCOPES)
                creds = flow.run_local_server()
            # Save the credentials for the next run
            with open('./utils/token.pickle', 'wb') as token:
                pickle.dump(creds, token)

        return build('sheets', 'v4', credentials=creds)


    def get_coors(self, model_name, dataset_name, transfer, data_augmentation):
        def table_config_coor(i, rows, transfer, data_augmentation):
            vectors = []
            step = 3
            print(data_augmentation)

            if len(data_augmentation) == 0:
                if transfer:
                    vectors.append(np.array([1,0,0,0]))
                else:
                    vectors.append(np.array([0,0,0,0]))

            for data_aug_type in data_augmentation:
                vector = []
                if transfer:
                    vector = [1]
                else:
                    vector = [0]
                if data_aug_type == "s":
                    vector.extend([0,0,1])
                elif data_aug_type == "p":
                    vector.extend([0,1,0])
                elif data_aug_type == "c":
                    vector.extend([1,0,0])
                elif data_aug_type == "n":
                    vector.extend([0,0,0])
                vectors.append(np.array(vector))

            config_v = np.array([0,0,0,0])
            for v in vectors:
                config_v = np.logical_or(config_v == 1, v == 1).astype(np.int)
            print(config_v)

            for ix in range(8):
                ix = (i+2) + ix*step
                s_config_v = np.array([int(c) for c in rows[ix][1:5]])
                if np.array_equal(s_config_v,config_v):
                    break

            return ix

        def f(model_name):
            return model_name.lower().replace("_", "-")
        # Call the Sheets API
        sheet = self.service.spreadsheets()
        result = sheet.values().get(spreadsheetId=self.SAMPLE_SPREADSHEET_ID,
                                    range=self.SAMPLE_RANGE_NAME).execute()
        values = result.get('values', [])

        if not values:
            print('No data found.')
        else:
            for i, row in enumerate(values):
                if len(row) == 1:
                    if f(row[0]) == f(model_name):
                        break
            row = table_config_coor(i, values, transfer, data_augmentation)+1

            for j, col in enumerate(values[i+1]):
                if f(col) == f(dataset_name):
                    break

            col = chr(65+j)

        return row, col

    def update_cell(self, col, row, text):
        values = [[text],]
        body = {"values":values}
        result = self.service.spreadsheets().values().update(
                        spreadsheetId=self.SAMPLE_SPREADSHEET_ID,
                        range="{}!{}{}".format(self.SAMPLE_RANGE_NAME, col, row),
                        valueInputOption="USER_ENTERED", body=body).execute()

    def init_exp(self, machine_name, log_path):
        self.update_cell(self.col, self.row, machine_name)
        self.update_cell(self.col, self.row+2, "{} {}".format(log_path, machine_name))

    def end_exp(self, avrg, acc_list):
        self.update_cell(self.col, self.row, avrg)
        self.update_cell(self.col, self.row+1, acc_list)

class GDriveCrossKfold(GDrive):
    def __init__(self, model_name, dataset_list, transfer, data_augmentation):
        self.SAMPLE_SPREADSHEET_ID = '1GECoXg8hU2V1VIvbGdqcZDzcsi16zynRq56L72F6xkM'
        self.SAMPLE_RANGE_NAME = 'auto-cross-kfold'

        self.service = self.get_service()
        self.row, self.col = self.get_coors(model_name, dataset_list, transfer, data_augmentation)

        self.dataset_list = dataset_list
        self.map_trick = {
            "CK": 0,
            "MMI": 1,
            "OULU": 2,
            "MUG": 3,
            "AFEW": 4
        }


    def get_coors(self, model_name, dataset_list, transfer, data_augmentation):
        def table_config_coor(i, rows, transfer, data_augmentation):
            vectors = []
            step = 3
            print(data_augmentation)

            if len(data_augmentation) == 0:
                if transfer:
                    vectors.append(np.array([1,0,0,0]))
                else:
                    vectors.append(np.array([0,0,0,0]))

            for data_aug_type in data_augmentation:
                vector = []
                if transfer:
                    vector = [1]
                else:
                    vector = [0]
                if data_aug_type == "s":
                    vector.extend([0,0,1])
                elif data_aug_type == "p":
                    vector.extend([0,1,0])
                elif data_aug_type == "c":
                    vector.extend([1,0,0])
                elif data_aug_type == "n":
                    vector.extend([0,0,0])
                vectors.append(np.array(vector))

            config_v = np.array([0,0,0,0])
            for v in vectors:
                config_v = np.logical_or(config_v == 1, v == 1).astype(np.int)
            print(config_v)

            for ix in range(8):
                ix = (i+2) + ix*step
                s_config_v = np.array([int(c) for c in rows[ix][1:5]])
                if np.array_equal(s_config_v,config_v):
                    break

            return ix

        def f(model_name):
            return model_name.lower().replace("_", "-")
        # Call the Sheets API
        sheet = self.service.spreadsheets()
        result = sheet.values().get(spreadsheetId=self.SAMPLE_SPREADSHEET_ID,
                                    range=self.SAMPLE_RANGE_NAME).execute()
        values = result.get('values', [])

        if not values:
            print('No data found.')
        else:
            for i, row in enumerate(values):
                if len(row) == 1:
                    if f(row[0]) == f(model_name):
                        break
            row = table_config_coor(i, values, transfer, data_augmentation)+1

            # for j, col in enumerate(values[i+1]):
            #     if f(col) == f(dataset_name):
            #         break

            col = 6

        return row, col

    def init_exp(self, machine_name, log_path):
        def f(col):
            mod = col % 26
            res = col // 26
            if res > 0:
                col_l = chr(65+res-1)
            else:
                col_l = ""
            col_l += chr(65+mod)

            return col_l

        self.map_cols = {}

        for dataset_name in self.dataset_list:
            self.map_cols[dataset_name] = f(self.col + self.map_trick[dataset_name])
            self.update_cell(self.map_cols[dataset_name], self.row, machine_name)
            self.update_cell(self.map_cols[dataset_name], self.row+2, "{} {}".format(log_path, machine_name))


    def end_exp(self, acc_map):
        for dataset_name, acc_list in acc_map.items():
            acc = str(np.mean(acc_list)).replace('.',',')
            self.update_cell(self.map_cols[dataset_name], self.row, acc)
            self.update_cell(self.map_cols[dataset_name], self.row+1, str(acc_list))


class GDrive1VAll(GDrive):
    def __init__(self, model_name, dataset_name, test_dataset_list, transfer, data_augmentation):
        self.SAMPLE_SPREADSHEET_ID = '1GECoXg8hU2V1VIvbGdqcZDzcsi16zynRq56L72F6xkM'
        self.SAMPLE_RANGE_NAME = 'auto-1-vs-all'
        self.dataset_name = dataset_name
        self.test_dataset_list = test_dataset_list
        self.service = self.get_service()
        self.row, self.col = self.get_coors(model_name, dataset_name, test_dataset_list, transfer, data_augmentation)

        self.map_trick = {
            "CK": 0,
            "MMI": 1,
            "OULU": 2,
            "MUG": 3,
            "AFEW": 4
        }

    def get_coors(self, model_name, dataset_name, test_dataset_list, transfer, data_augmentation):
        def f(model_name):
            return model_name.lower().replace("_", "-")

        def table_config_coor(i, rows, transfer, data_augmentation):
            vectors = []
            step = 3
            shift = 4
            print(data_augmentation)

            if len(data_augmentation) == 0:
                if transfer:
                    vectors.append(np.array([1,0,0,0]))
                else:
                    vectors.append(np.array([0,0,0,0]))

            for data_aug_type in data_augmentation:
                vector = []
                if transfer:
                    vector = [1]
                else:
                    vector = [0]
                if data_aug_type == "s":
                    vector.extend([0,0,1])
                elif data_aug_type == "p":
                    vector.extend([0,1,0])
                elif data_aug_type == "c":
                    vector.extend([1,0,0])
                elif data_aug_type == "n":
                    vector.extend([0,0,0])
                vectors.append(np.array(vector))

            config_v = np.array([0,0,0,0])
            for v in vectors:
                config_v = np.logical_or(config_v == 1, v == 1).astype(np.int)
            print(config_v)

            for iy in range(8):
                iy = iy*4 + shift + iy*step
                s_config_v = np.array([int(c) for c in rows[i][iy:iy+4]])
                if np.array_equal(s_config_v,config_v):
                    break

            return iy

        sheet = self.service.spreadsheets()
        result = sheet.values().get(spreadsheetId=self.SAMPLE_SPREADSHEET_ID,
                                    range=self.SAMPLE_RANGE_NAME).execute()
        values = result.get('values', [])

        if not values:
            print('No data found.')
        else:
            for i, row in enumerate(values):
                if len(row) > 0:
                    if f(row[1]) == f(model_name):
                        break

            col = table_config_coor(i, values, transfer, data_augmentation)

            for j in range(i+2, i+2+5):
                if f(values[j][col-1]) == f(dataset_name):
                    break
            row = j+1
        return row, col



    def init_exp(self, machine_name, log_path):
        def f(col):
            mod = col % 26
            res = col // 26
            if res > 0:
                col_l = chr(65+res-1)
            else:
                col_l = ""
            col_l += chr(65+mod)

            return col_l

        self.map_cols = {}
        for dataset_name in (self.test_dataset_list + [self.dataset_name]):
            self.map_cols[dataset_name] = f(self.col + self.map_trick[dataset_name])

        for dataset_name in self.test_dataset_list:
            self.update_cell(self.map_cols[dataset_name], self.row, machine_name)

        self.update_cell(self.map_cols[self.dataset_name], self.row, "{} {}".format(log_path, machine_name))


    def end_exp(self, acc_map):
        for dataset_name, acc in acc_map.items():
            self.update_cell(self.map_cols[dataset_name], self.row, acc)

# models = ["c3d-block-lstm", "VGG16-lstm", "C3D", "Inceptionv3-lstm", "ResNet3D-101", "I3D", "ResNet101-lstm", "ResNet18-lstm", "ResNet3D-18"]
# datasets = ["CK", "MMI", "OULU", "MUG", "AFEW"]
# das = ["n", "c", "p", "s"]
# pres = [True]
# #
# acc_map = {
#     "CK": 0.231,
#     "MMI": 0.3213,
#     "OULU": 0.2131,
#     "MUG": 0.9123,
#     "AFEW": 0.123
# }
# for m in models:
#     for dt in datasets:
#         test_list = []
#         for d in datasets:
#             if d == dt:
#                 continue
#             test_list.append(d)
#
#         for da in das:
#             for pre in pres:
#                 print("{}-{}-{}-{}-{}".format(m, dt, test_list,  pre, da))
#                 gd = GDrive1VAll(m, dt, test_list, pre, da)
#                 gd.init_exp("dl-19", "{}-{}-{}-{}-{}".format(m, dt, test_list,  pre, da))
#
#                 gd.end_exp(acc_map)
#         from time import sleep
#         sleep(30)

# gd = GDrive("c3d", "CK", True, "s")
# gd.init_exp("dl-19", "asda  asdasd asd asd asda sd BIG LOG/LOGPATH")
# from time import sleep
# sleep(20)
#
# gd.end_exp("0.2312312", "[0.23123, 0.123123, 0.123123, 0.213123, 0.123123]")
