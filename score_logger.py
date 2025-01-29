from statistics import mean
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import deque
import os
import csv
import numpy as np


class ScoreLogger:
    def __init__(self, name, slices):
        self.scores = deque(maxlen=None)
        self.name = name
        self.csv = f'./scores/{name}_scores.csv'
        self.png = f'./scores/{name}_scores.png'
        self.slices = slices

        if os.path.exists(self.png):
            os.remove(self.png)
        if os.path.exists(self.csv):
            os.remove(self.csv)

    def add_score(self, score, episode):
        self._save_csv(self.csv, score)
        if episode % self.slices == 0 and episode != 0:
            self._save_png(input_path=self.csv,
                        output_path=self.png,
                        x_label="messages",
                        y_label="average cost",
                        show_legend=True)
        self.scores.append(score)
        mean_score = mean(self.scores)
        print(f'Cost: (min: {min(self.scores)}, avg: {mean_score}, max: {max(self.scores)}) Message: {episode}\n')

    def _save_png(self, input_path, output_path, x_label, y_label, show_legend):
        x = []
        y = []
        with open(input_path, "r") as scores:
            reader = csv.reader(scores)
            data = list(reader)
            input_lst = []
            for value in data:
                input_lst.append(int(value[0]))
            output_lst = self.calculate_means(input_lst, self.slices)
            for i in range(0, len(output_lst)):
                x.append(int(i * self.slices)+1000)
                y.append(int(output_lst[i]))

        #plt.subplots()
        #plt.plot(x, y, marker='o', linestyle='dashed', label="score per message")

        plt.plot(x, y, marker='o', linestyle="dashed", label="average")

        plt.title(self.name)
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        if show_legend:
            plt.legend(loc="upper left")

        plt.savefig(output_path, bbox_inches="tight")
        plt.close()

    def _save_csv(self, path, score):
        if not os.path.exists(path):
            with open(path, "w", newline=''):
                pass
        scores_file = open(path, "a", newline='')
        with scores_file:
            writer = csv.writer(scores_file)
            writer.writerow([score])

    def calculate_means(self, original_list, window_size=1000):
        means_list = []
        num_chunks = len(original_list) // window_size

        for i in range(num_chunks):
            start_idx = i * window_size
            end_idx = start_idx + window_size
            chunk = original_list[start_idx:end_idx]
            mean = sum(chunk) / len(chunk)
            means_list.append(mean)

        return means_list