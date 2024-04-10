import numpy as np
import pandas as pd
from sklearn.utils import shuffle

classes = {
    "Normal": 0,
    "Fuzzers": 1,
    "Analysis": 2,
    "Backdoor": 3,
    "DoS": 4,
    "Exploits": 5,
    "Generic": 6,
    "Reconnaissance": 7,
    "Shellcode": 8,
    "Worms": 9
}

proto = {
    'sccopmce': 0, 'ttp': 1, 'ggp': 2, 'isis': 3, 'pim': 4, 'idpr': 5, 'sun-nd': 6, 'crtp': 7, 'ipcv': 8, 'bna': 9, 
    'irtp': 10, 'compaq-peer': 11, 'sm': 12, 'ipx-n-ip': 13, 'sat-mon': 14, 'ifmp': 15, 'netblt': 16, 'nsfnet-igp': 17, 
    'cphb': 18, 'qnx': 19, 'xnet': 20, 'fc': 21, 'skip': 22, 'xtp': 23, 'il': 24, 'idpr-cmtp': 25, 'smp': 26, 
    'ipv6-frag': 27, 'ipv6-route': 28, 'ipv6-opts': 29, 'encap': 30, 'mhrp': 31, 'aes-sp3-d': 32, 'st2': 33, 'mux': 34, 
    'arp': 35, 'unas': 36, 'idrp': 37, 'ipv6': 38, 'narp': 39, 'pnni': 40, 'sat-expak': 41, 'br-sat-mon': 42, 'trunk-2': 43, 
    'xns-idp': 44, 'micp': 45, 'cpnx': 46, 'cbt': 47, 'l2tp': 48, 'pipe': 49, 'aris': 50, 'iso-tp4': 51, 'merit-inp': 52, 
    'rvd': 53, 'sdrp': 54, 'wb-mon': 55, 'vines': 56, 'iatp': 57, 'ospf': 58, 'zero': 59, 'egp': 60, 'igp': 61, 'ax.25': 62, 
    'ipip': 63, 'iplt': 64, 'sps': 65, 'tcf': 66, 'gre': 67, 'scps': 68, 'pvp': 69, 'ippc': 70, 'secure-vmtp': 71, 'eigrp': 72, 
    'srp': 73, 'sctp': 74, 'argus': 75, 'trunk-1': 76, 'kryptolan': 77, 'cftp': 78, 'pgm': 79, 'ipv6-no': 80, 'stp': 81, 'ptp': 82, 
    'wb-expak': 83, 'mobile': 84, 'gmtp': 85, 'pri-enc': 86, 'tcp': 87, 'icmp': 88, 'ipnip': 89, 'pup': 90, 'vrrp': 91, 'tp++': 92, 
    'leaf-1': 93, 'rsvp': 94, 'udp': 95, '3pc': 96, 'ip': 97, 'rdp': 98, 'fire': 99, 'visa': 100, 'mfe-nsp': 101, 'dcn': 102, 
    'sprite-rpc': 103, 'vmtp': 104, 'iso-ip': 105, 'tlsp': 106, 'crudp': 107, 'uti': 108, 'dgp': 109, 'igmp': 110, 'hmp': 111, 
    'prm': 112, 'sep': 113, 'bbn-rcc': 114, 'larp': 115, 'any': 116, 'etherip': 117, 'a/n': 118, 'snp': 119, 'mtp': 120, 'i-nlsp': 121, 
    'rtp': 122, 'chaos': 123, 'wsn': 124, 'ddx': 125, 'emcon': 126, 'nvp': 127, 'swipe': 128, 'ib': 129, 'leaf-2': 130, 'ddp': 131, 
    'ipcomp': 132
}

service = {
    'http': 0, 'ftp': 1, 'radius': 2, 'dhcp': 3, 'snmp': 4, 'ssl': 5, 'pop3': 6, '-': 7, 'irc': 8, 'ftp-data': 9, 'ssh': 10, 'dns': 11, 'smtp': 12
}

state = {
    'PAR': 0, 'RST': 1, 'URN': 2, 'REQ': 3, 'ECO': 4, 'no': 5, 'CON': 6, 'FIN': 7, 'INT': 8, 'CLO': 9, 'REQ': 10, 'RST': 11, 'INT': 12, 'ACC': 13
}

class Data():
    def __init__(self, training_file, testing_file):
        self.training_file = training_file
        self.testing_file = testing_file
        #Create basic train test split and shuffle data
        self.X_train, self.y_train, self.X_test, self.y_test = self.process_data()
        #Create normalized data
        self.X_train_norm, self.X_test_norm = self.normalize_data(self.X_train, self.X_test)
        #Create standardized data
        self.X_train_std, self.X_test_std = self.standardize_data(self.X_train, self.X_test)
        #Create standardized normalized data
        self.X_train_std_norm, self.X_test_std_norm = self.standardize_data(self.X_train_norm, self.X_test_norm)

    def process_data(self):
        #Makes all values floats and shuffles the data then splits it for testing and training
        training_data = load_data(self.training_file)
        testing_data = load_data(self.testing_file)

        training_data = shuffle(training_data)
        testing_data = shuffle(testing_data)
        
        #Shuffle again to randomize data more
        training_data = shuffle(training_data)
        testing_data = shuffle(testing_data)

        X_train, y_train = split_data(training_data)
        X_test, y_test = split_data(testing_data)

        return X_train, y_train, X_test, y_test
    
    def normalize_data(self, X_train, X_test):
        X_train_norm = normalize(X_train)
        X_test_norm = normalize(X_test)
        return X_train_norm, X_test_norm
    
    def standardize_data(self, X_train, X_test):
        X_train_std = standardize(X_train)
        X_test_std = standardize(X_test)
        return X_train_std, X_test_std

def load_data(file_path):
    df = pd.read_csv(file_path)
    df['attack_cat'] = df['attack_cat'].map(classes)
    df['proto'] = df['proto'].map(proto)
    df['service'] = df['service'].map(service)
    df['state'] = df['state'].map(state)
    return df

def split_data(data):
    X_data = data.iloc[:, 1:-2].astype(np.float32).values.tolist()
    y_data = data.iloc[:, -1].astype(np.float32).values.tolist()
    
    return X_data, y_data

def standardize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X - mean) / std

def normalize(X):
    min_val = np.min(X, axis=0)
    max_val = np.max(X, axis=0)
    return(X - min_val) / (max_val - min_val)
