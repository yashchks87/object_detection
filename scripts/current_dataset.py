import pickle
import torch, torchvision
import multiprocessing as mp
import pandas as pd


def detect_single_channel(img_path):
    img = torchvision.io.read_file(img_path)
    img = torchvision.io.decode_jpeg(img)
    return (img_path, img.shape)
def drop_single_channel_imgs(csv_file):
    paths = csv_file['updated_paths'].values.tolist()
    issues = []
    with mp.Pool(6) as p:
        detected = list(p.map(detect_single_channel, paths))
    for x in detected:
        if x[1][0] == 1:
            issues.append(x[0])
    return csv_file[~csv_file['updated_paths'].isin(issues)]

def filter_face(face_score):
    if face_score != float('-inf'):
        return 1
    else:
        return 0

def filtered_face_score(csv_file):
    csv_file['is_face'] = csv_file['face_score'].apply(lambda x: filter_face(x))
    return csv_file


def get_dataset(csv_path, issue_path_pickle):
    wiki_csv = pd.read_csv(csv_path)
    with open(issue_path_pickle, 'rb') as handle:
        issues = pickle.load(handle)
    wiki_csv = wiki_csv[~wiki_csv['updated_paths'].isin(issues)]
    wiki_csv = drop_single_channel_imgs(wiki_csv)
    wiki_csv = filtered_face_score(wiki_csv)
    return wiki_csv