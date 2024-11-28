import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import pandas as pd

label_list = ["attention", "ready", "crank", "stop", "move",
              "turn-left", "turn-right", "speed-up",
              "slow-down", "kill", "standby"]

Vh = torch.Tensor([
        [-0.103361,-0.798408,-0.131954,0.352947,-0.081768,-0.412223,-0.070942,0.155816,-0.024347,0.017349,-0.018822,0.034797,0.000730,-0.031361,-0.000756,0.022539],
        [0.218639,-0.368440,0.284466,-0.739076,0.061756,-0.230715,0.141214,-0.326127,0.021192,-0.026500,0.018966,-0.036914,0.008191,-0.036255,0.005278,-0.040844],
        [0.900175,-0.070915,-0.125096,0.180153,0.315689,0.100923,-0.022589,0.133284,0.016236,0.048951,0.030029,0.047888,0.025333,0.024433,0.025562,0.026156],
        [-0.038412,-0.114323,-0.770893,-0.450624,-0.113651,0.225537,-0.276050,0.116383,-0.098087,0.070340,-0.068054,0.085036,-0.056386,0.021671,-0.054926,0.038558],
        [0.066672,0.230998,-0.358444,0.080609,0.080965,-0.448847,-0.125605,-0.290176,0.121264,-0.450253,0.066069,-0.464718,0.033864,-0.174678,0.030418,-0.169919],
        [-0.058899,-0.353633,0.074098,0.018155,-0.038150,0.640558,0.083460,0.086453,0.237292,-0.385181,0.202396,-0.406165,0.078472,-0.043144,0.085379,-0.116315],
        [-0.227879,0.035763,-0.204918,-0.091671,0.458679,-0.129627,0.249868,0.146523,0.371752,0.190057,0.463705,0.179026,0.280783,0.058801,0.284387,-0.027185],
        [-0.091214,0.076996,0.184972,-0.242359,0.329142,-0.183001,-0.136195,0.682239,-0.120635,-0.268953,-0.202045,-0.243419,-0.017558,0.071312,-0.025023,0.272015],
        [-0.231146,-0.147246,0.028801,0.120743,0.683964,0.202314,-0.328802,-0.446263,-0.139381,-0.005032,-0.152512,0.026322,-0.098045,0.135722,-0.102457,0.107457],
        [-0.088562,-0.023431,-0.104609,0.000024,0.282513,0.067118,0.440325,0.145689,-0.270202,0.020743,-0.237721,0.059333,-0.148945,-0.497822,-0.139380,-0.508258],
        [-0.007912,-0.026425,-0.240579,0.066649,-0.014073,0.032886,0.620249,-0.178565,-0.166598,-0.182886,-0.371379,-0.049812,0.201233,0.266545,0.192245,0.411025],
        [-0.007688,0.001088,-0.106493,0.014437,0.042857,-0.064925,0.322649,0.029512,0.138208,0.058772,0.266712,-0.172116,-0.553819,0.378741,-0.548771,0.052348],
        [0.007107,-0.002304,0.019164,-0.019957,-0.009999,-0.053937,-0.058295,0.078985,-0.012930,-0.106217,-0.226992,0.067211,0.103466,0.677941,0.125902,-0.657936],
        [0.009572,0.009664,0.014303,0.002571,-0.020748,-0.007644,0.017526,-0.006709,-0.530172,-0.528296,0.511207,0.415644,-0.029939,0.057403,0.005786,0.016338],
        [0.004816,0.006609,-0.000411,-0.008599,0.001855,0.000002,-0.008261,0.005669,0.586830,-0.452037,-0.300886,0.549591,-0.135643,-0.109473,-0.156881,0.058904],
        [-0.000359,-0.000556,-0.000504,0.000783,-0.001752,0.004352,0.002421,0.003169,-0.019934,-0.005481,0.019014,0.004362,0.707567,0.009846,-0.705731,-0.018675]
    ])

def extract_data(gesture_data_csv_path: str):
    data = pd.read_csv(gesture_data_csv_path)
    seq_len = 15
    seq_overlap = int(seq_len / 1)
    sequences = torch.tensor([])
    labels = torch.tensor([])
    for seq_idx in range(0, (len(data['gesture']) - seq_len), seq_overlap):
        sequence = torch.tensor([])
        label = torch.zeros(size=(1, 11))
        for idx in range(seq_idx, seq_idx + seq_len):
            gesture = torch.tensor([[[
                data['left_wrist_x'].iloc[idx],
                data['left_wrist_y'].iloc[idx],
                data['left_elbow_x'].iloc[idx],
                data['left_elbow_y'].iloc[idx],
                data['right_elbow_x'].iloc[idx],
                data['right_elbow_y'].iloc[idx],
                data['right_wrist_x'].iloc[idx],
                data['right_wrist_y'].iloc[idx]
            ]]], dtype=torch.float32)
            sequence = torch.cat(tensors=(sequence, gesture), dim=1)
    
        # One-Hot Encoding
        label[0, int(data['gesture'].iloc[idx])] = 1.0
    
        # Concatenate Tensors
        sequences = torch.cat(tensors=(sequences, sequence), dim=0)
        labels = torch.cat(tensors=(labels, label), dim=0)

    return (labels, sequences)

def extract_pca_data(gesture_data_csv_path: str):
    seq_len = 15
    data = pd.read_csv(gesture_data_csv_path)
    A = torch.Tensor(data[[
        'right_wrist_x',
        'right_wrist_y',
        'left_wrist_x',
        'left_wrist_y',
        'right_elbow_x',
        'right_elbow_y',
        'left_elbow_x',
        'left_elbow_y',
        'right_hip_x',
        'right_hip_y',
        'left_hip_x',
        'left_hip_y',
        'right_shoulder_x',
        'right_shoulder_y',
        'left_shoulder_x',
        'left_shoulder_y'
    ]].to_numpy())
    seq_overlap = int(seq_len / 1)
    sequences = torch.tensor([])
    labels = torch.tensor([])
    for seq_idx in range(0, (len(data['gesture']) - seq_len), seq_overlap):
        sequence = torch.tensor([])
        label = torch.zeros(size=(1, 11))
        for idx in range(seq_idx, seq_idx + seq_len):
            gesture = A[idx, :]
            pca_gesture = torch.Tensor([[[0, 0, 0, 0]]])
            pca_gesture[0, 0, :] = torch.matmul(gesture, Vh[:4, :].T)
            sequence = torch.cat(tensors=(sequence, pca_gesture), dim=1)
    
        # One-Hot Encoding
        label[0, int(data['gesture'].iloc[idx])] = 1.0
    
        # Concatenate Tensors
        sequences = torch.cat(tensors=(sequences, sequence), dim=0)
        labels = torch.cat(tensors=(labels, label), dim=0)

    return (labels, sequences)

class GestureDataset(Dataset):
    def __init__(self, pca=False):
        files = [
            "images/chris_attention/gesture_data.csv",
            "images/chris_crank/gesture_data.csv",
            "images/chris_forward/gesture_data.csv",
            "images/chris_left/gesture_data.csv",
            "images/chris_right/gesture_data.csv",
            "images/chris_ready/gesture_data.csv",
            "images/chris_slowdown/gesture_data.csv",
            "images/chris_speedup/gesture_data.csv",
            "images/chris_stop/gesture_data.csv",
            "images/chris_kill/gesture_data.csv",
            "images/janie_attention_001/gesture_data.csv",
            "images/janie_crank_001/gesture_data.csv",
            "images/janie_kill_001/gesture_data.csv",
            "images/janie_left_001/gesture_data.csv",
            "images/janie_move_001/gesture_data.csv",
            "images/janie_move_002/gesture_data.csv",
            "images/janie_ready_001/gesture_data.csv",
            "images/janie_right_001/gesture_data.csv",
            "images/janie_slowdown_001/gesture_data.csv",
            "images/janie_speedup_001/gesture_data.csv",
            "images/janie_standby_001/gesture_data.csv",
            "images/janie_stop_001/gesture_data.csv",
            "images/stephen_attention_001/gesture_data.csv",
            "images/stephen_attention_002/gesture_data.csv",
            "images/stephen_attention_003/gesture_data.csv",
            "images/stephen_attention_004/gesture_data.csv",
            "images/stephen_crank_001/gesture_data.csv",
            "images/stephen_crank_002/gesture_data.csv",
            "images/stephen_crank_003/gesture_data.csv",
            "images/stephen_forward_001/gesture_data.csv",
            "images/stephen_kill_001/gesture_data.csv",
            "images/stephen_left_001/gesture_data.csv",
            "images/stephen_ready_001/gesture_data.csv",
            "images/stephen_right_001/gesture_data.csv",
            "images/stephen_slowdown_001/gesture_data.csv",
            "images/stephen_slowdown_002/gesture_data.csv",
            "images/stephen_slowdown_003/gesture_data.csv",
            "images/stephen_speedup_001/gesture_data.csv",
            "images/stephen_speedup_002/gesture_data.csv",
            "images/stephen_standby_001/gesture_data.csv",
            "images/stephen_stop_001/gesture_data.csv",
            "images/stephen_standby_crank_001/gesture_data.csv",
            "images/stephen_standby_stop_001/gesture_data.csv",
            "images/stephen_standby_left_001/gesture_data.csv",
            "images/stephen_standby_right_001/gesture_data.csv",
            "images/stephen_attention_stop_001/gesture_data.csv",
            "images/john_attention_001/gesture_data.csv",
            "images/john_crank_001/gesture_data.csv",
            "images/john_kill_001/gesture_data.csv",
            "images/john_left_001/gesture_data.csv",
            "images/john_move_001/gesture_data.csv",
            "images/john_ready_001/gesture_data.csv",
            "images/john_right_001/gesture_data.csv",
            "images/john_slowdown_001/gesture_data.csv",
            "images/john_slowdown_002/gesture_data.csv",
            "images/john_speedup_001/gesture_data.csv",
            "images/john_standby_001/gesture_data.csv",
            "images/john_standby_002/gesture_data.csv",
            "images/john_stop_001/gesture_data.csv",
            "images/terri_attention_001/gesture_data.csv",
            "images/terri_crank_001/gesture_data.csv",
            "images/terri_kill_001/gesture_data.csv",
            "images/terri_left_001/gesture_data.csv",
            "images/terri_move_001/gesture_data.csv",
            "images/terri_ready_001/gesture_data.csv",
            "images/terri_right_001/gesture_data.csv",
            "images/terri_slowdown_001/gesture_data.csv",
            "images/terri_slowdown_002/gesture_data.csv",
            "images/terri_speedup_001/gesture_data.csv",
            "images/terri_speedup_002/gesture_data.csv",
            "images/terri_standby_001/gesture_data.csv",
            "images/terri_standby_002/gesture_data.csv",
            "images/terri_stop_001/gesture_data.csv",
            "images/terri_stop_001/gesture_data.csv",
        ]

        labels = torch.Tensor([])
        sequences = torch.Tensor([])

        for file in files:
            if pca:
                (label, sequence) = extract_pca_data(file)
            else:
                (label, sequence) = extract_data(file)
            labels = torch.cat(tensors=(labels, label), dim=0)
            sequences = torch.cat(tensors=(sequences, sequence), dim=0)
    
        self.labels = labels
        self.sequences = sequences
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx, :, :], self.labels[idx, :]

class GestureAccuracy():
    def __init__(self):
        self.accuracy_dict = self._get_empty_accuracy_dict()
    
    def reset(self):
        self.accuracy_dict = self._get_empty_accuracy_dict()
    
    def update_accuracy(self, pred, actual):
        batch_size = pred.shape[0]
        for idx in range(batch_size):
            pred_idx = torch.argmax(pred[idx, :])
            actual_idx = torch.argmax(actual[idx, :])
            for key in self.accuracy_dict.keys():
                key_idx = label_list.index(key)
                if pred_idx == actual_idx and pred_idx == key_idx:
                    # True Positive
                    self.accuracy_dict[key]["true_positive"] += 1
                    pass
                elif pred_idx == key_idx:
                    # False Positive
                    self.accuracy_dict[key]["false_positive"] += 1
                    pass
                elif actual_idx == key_idx:
                    # False Negative
                    self.accuracy_dict[key]["false_negative"] += 1
                    pass
                else:
                    # True Negative
                    self.accuracy_dict[key]["true_negative"] += 1

    def accuracy_report(self):
        print("\nAccuracy Report:")
        avg_accuracy = 0
        avg_recall = 0
        avg_precision = 0
        avg_f1_score = 0
        cnt = 0
        for key in self.accuracy_dict.keys():
            tp = self.accuracy_dict[key]["true_positive"]
            tn = self.accuracy_dict[key]["true_negative"]
            fp = self.accuracy_dict[key]["false_positive"]
            fn = self.accuracy_dict[key]["false_negative"]

            if (tp + tn + fp + fn):
                accuracy = ((tp + tn) / (tp + tn + fp + fn))
            else:
                accuracy = 0.0
            
            if (fp + tp) > 0:
                precision = ( tp / (fp + tp))
            else:
                precision = 0.0
            
            if (fn + tp) > 0.0:
                recall = ( tp / (fn + tp) )
            else:
                recall = 0.0

            if (precision + recall) > 0.0:
                f1_score = 2 * precision * recall / (precision + recall)
            else:
                f1_score = 0.0
            
            avg_accuracy += accuracy
            avg_precision += precision
            avg_recall += recall
            avg_f1_score += f1_score
            cnt += 1
            print(f"[{key}] Accuracy: {accuracy:.3f} | Precision {precision:.3f} | Recall: {recall:.3f} | F1-Score: {f1_score:.3f}")
        avg_accuracy /= cnt
        avg_precision /= cnt
        avg_recall /= cnt
        avg_f1_score /= cnt
        print(f"\n[AVERAGE] Accuracy: {avg_accuracy:.3f} | Precision {avg_precision:.3f} | Recall: {avg_recall:.3f} | F1-Score: {avg_f1_score:.3f}\n\n")

    def _get_empty_accuracy_dict(self):
        accuracy_dict = {
            "attention": {
                "true_positive": 0,
                "true_negative": 0,
                "false_positive": 0,
                "false_negative": 0
            },
            "ready": {
                "true_positive": 0,
                "true_negative": 0,
                "false_positive": 0,
                "false_negative": 0
            },
            "crank": {
                "true_positive": 0,
                "true_negative": 0,
                "false_positive": 0,
                "false_negative": 0
            },
            "stop":  {
                "true_positive": 0,
                "true_negative": 0,
                "false_positive": 0,
                "false_negative": 0
            },
            "move": {
                "true_positive": 0,
                "true_negative": 0,
                "false_positive": 0,
                "false_negative": 0
            },
            "turn-left": {
                "true_positive": 0,
                "true_negative": 0,
                "false_positive": 0,
                "false_negative": 0
            },
            "turn-right": {
                "true_positive": 0,
                "true_negative": 0,
                "false_positive": 0,
                "false_negative": 0
            },
            "speed-up": {
                "true_positive": 0,
                "true_negative": 0,
                "false_positive": 0,
                "false_negative": 0
            },
            "slow-down": {
                "true_positive": 0,
                "true_negative": 0,
                "false_positive": 0,
                "false_negative": 0
            },
            "kill": {
                "true_positive": 0,
                "true_negative": 0,
                "false_positive": 0,
                "false_negative": 0
            },
            "standby": {
                "true_positive": 0,
                "true_negative": 0,
                "false_positive": 0,
                "false_negative": 0
            }
        }
        return accuracy_dict

if __name__ == "__main__":
    # dataset = GestureDataset()
    # print(dataset.labels.shape)
    # print(dataset.sequences.shape)

    accuracy = GestureAccuracy()
    accuracy.update_accuracy(
        pred = torch.Tensor([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ]),
        actual = torch.Tensor([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ])
    )

    accuracy.accuracy_report()