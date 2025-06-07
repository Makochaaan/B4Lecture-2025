#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
B4輪講最終課題 パターン認識に挑戦してみよう
ベースラインスクリプト
特徴量；周波数領域における分布(MFCC)
識別器；HMM
"""


from __future__ import division
from __future__ import print_function

import argparse

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hmmlearn import hmm
from keras.utils import np_utils
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

def my_HMM(n_components, covariance_type="full", n_iter=1000):
    """
    HMMモデルの構築
    Args:
        n_components: 隠れ状態の数
        covariance_type: 共分散行列のタイプ ("full", "diag", "tied", "spherical")
        n_iter: 学習のイテレーション回数
    Returns:
        model: 定義済みモデル
    """
    model = hmm.GaussianHMM(n_components=n_components, covariance_type=covariance_type, n_iter=n_iter)
    return model

def feature_extraction(path_list):
    """
    wavファイルのリストから特徴抽出を行い，リストで返す
    扱う特徴量はMFCC13次元の平均（0次は含めない）
    Args:
        path_list: 特徴抽出するファイルのパスリスト
    Returns:
        features: 特徴量
    """
    features_list = []
    # 各音声ファイルからMFCCを抽出し、リストに追加
    # mfccの平均をaxis=1で計算した場合、各音声においての特徴量が削られてしまう。
    # そのため、MFCCの時系列をそのままリストに追加し、削減が行われないように利用する
    # ここで得たMFCCは、ラベルごとにまとめてHMMの学習材料となる
    for path in path_list:
        y, sr = librosa.load(path)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T # .Tで (フレーム数, MFCC次元) に変換
        features_list.append(mfcc)
    
    return features_list


def plot_confusion_matrix(predict, ground_truth, title=None, cmap=plt.cm.Blues):
    """
    予測結果の混合行列をプロット
    Args:
        predict: 予測結果
        ground_truth: 正解ラベル
        title: グラフタイトル
        cmap: 混合行列の色
    Returns:
        Nothing
    """
    cm = confusion_matrix(predict, ground_truth)
    print(cm)
    plt.figure()
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.ylabel("Predicted")
    plt.xlabel("Ground truth")
    plt.show()


def write_result(paths, outputs):
    """
    結果をcsvファイルで保存する
    Args:
        paths: テストする音声ファイルリスト
        outputs:
    Returns:
        Nothing
    """

    with open("result.csv", "w") as f:
        f.write("path,output\n")
        assert len(paths) == len(outputs)
        for path, output in zip(paths, outputs):
            f.write("{path},{output}\n".format(path=path, output=output))


def main(best_n_components):
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_truth", type=str, help='テストデータの正解ファイルCSVのパス')
    args = parser.parse_args()

    # データの読み込み
    training = pd.read_csv("training.csv")
    test = pd.read_csv("test.csv")

    # 学習データの特徴抽出 (時系列MFCC)
    X_train = feature_extraction(training["path"].values)
    X_test = feature_extraction(test["path"].values)

    # 正解ラベルをone-hotベクトルに変換
    Y_train = np_utils.to_categorical(y=training["label"], num_classes=10)

    # 学習データを学習データとバリデーションデータに分割 (バリデーションセットを40%とした例)
    X_train, X_validation, Y_train, Y_validation = train_test_split(
        X_train, Y_train,
        test_size=0.4,
        random_state=20200616,
    )

    # クラスごとにMFCCシーケンスをまとめる
    features_by_label_concatenated = [[] for _ in range(10)]
    lengths_by_label = [[] for _ in range(10)]
    for mfcc_seq, label_onehot in zip(X_train, Y_train):
        label_idx = np.argmax(label_onehot)
        features_by_label_concatenated[label_idx].append(mfcc_seq)
        lengths_by_label[label_idx].append(len(mfcc_seq))

    models = []

    # 1回目の実行時に最適な状態数を決定する
    if best_n_components is None:
        # 状態数の候補
        n_components_candidates = [3, 4, 5, 6, 7, 8, 9, 10]
        best_n_components = []
        for label_idx in range(10):
            X = np.vstack(features_by_label_concatenated[label_idx])
            lengths = lengths_by_label[label_idx]
            X_val = [x for x, y in zip(X_validation, np.argmax(Y_validation, axis=1)) if y == label_idx]

            best_score = -np.inf
            best_n = n_components_candidates[0]
            best_model = None

            for n in n_components_candidates:
                model = my_HMM(n_components=n, covariance_type="full")
                model.fit(X, lengths)
                val_scores = [model.score(x) for x in X_val] if X_val else [0]
                mean_score = np.mean(val_scores)
                if mean_score > best_score:
                    best_score = mean_score
                    best_n = n
                    best_model = model

            best_n_components.append(best_n)
            models.append(best_model)
            print(f"label {label_idx}: best_n_components={best_n}")

        print("最適な状態数:", best_n_components)
    else:
        # 2回目以降は最適な状態数が与えられているので、それを使用
        for label_idx in range(10):
            X = np.vstack(features_by_label_concatenated[label_idx])
            lengths = lengths_by_label[label_idx]
            model = my_HMM(n_components=best_n_components[label_idx], covariance_type="full")
            model.fit(X, lengths)
            models.append(model)

    # バリデーションセットによるモデルの評価
    validation_predicts = []
    validation_truth = np.argmax(Y_validation, axis=1) # 正解ラベルは元のone-hotから変換

    for x_val_seq in X_validation:
        scores = []
        for m in models:
            # hmm.scoreは一つのシーケンスに対して評価
            scores.append(m.score(x_val_seq))
        
        # 最もスコアの高いモデルのインデックスを予測結果とする
        validation_predicts.append(np.argmax(scores))

    validation_predicts = np.array(validation_predicts)
    
    # バリデーション予測結果の保存
    with open("validation_predicts.csv", "w") as f:
        f.write("path,output,truth\n")
        for path, output, truth in zip(training["path"].values[len(X_train):], validation_predicts, validation_truth):
            f.write(f"{path},{output},{truth}\n")

    plot_confusion_matrix(validation_predicts, validation_truth, title="Validation Confusion Matrix")
    print("Validation accuracy: ", accuracy_score(validation_truth, validation_predicts))


    # 予測結果 (テストデータ)
    predicted_values = []
    for x_test_seq in X_test:
        scores = []
        for m in models:
            scores.append(m.score(x_test_seq))
        predicted_values.append(np.argmax(scores))

    # テストデータに対して推論した結果の保存
    write_result(test["path"].values, predicted_values)

    # テストデータに対する正解ファイルが指定されていれば評価を行う（accuracyと混同行列）
    if args.path_to_truth:
        test_truth = pd.read_csv(args.path_to_truth)
        truth_values = test_truth['label'].values
        plot_confusion_matrix(predicted_values, truth_values, title="Test Confusion Matrix")
        print("Test accuracy: ", accuracy_score(truth_values, predicted_values))

    return best_n_components

if __name__ == "__main__":
    best_n_components = None
    for n in range(5):
        print(f"{n+1}回目")
        print("========================================")
        best_n_components = main(best_n_components=best_n_components)
