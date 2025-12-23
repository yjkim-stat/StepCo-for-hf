import warnings
warnings.filterwarnings('ignore')
import os
import json

import tiktoken
from datasets import load_dataset


def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding_name = "cl100k_base"
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


class DataLoader:
    def __init__(self, dataset_name, root_path):
        self.dataset_name = dataset_name
        self.root_path = root_path
        # self.dataset_path = os.path.join(self.root_path, f'{dataset_name}.json')
        self.read_dataset()
        # self.print_dataset_info()

    def read_dataset(self):
        if self.dataset_name == 'amc23':
            self.dataset = load_dataset('math-ai/amc23')['test']
        elif self.dataset_name == 'olympiad':
            self.dataset = load_dataset('Hothan/OlympiadBench', 'OE_TO_maths_en_COMP')['train']
        elif self.dataset_name == 'math500':
            self.dataset = load_dataset('HuggingFaceH4/MATH-500')['test']
        elif self.dataset_name == 'minerva':
            self.dataset = load_dataset('math-ai/minervamath')['test']
        elif self.dataset_name == 'aime2024':
            self.dataset = load_dataset('Maxwell-Jia/AIME_2024')['train']
        elif self.dataset_name == 'aime2025':
            self.dataset = load_dataset('MathArena/aime_2025')['train']
        else:
            raise KeyError()
        # with open(self.dataset_path, 'r') as f:
        #     dataset = json.load(f)
        #     self.dataset = dataset

    def get_problems(self):
        # if self.dataset_name not in ['MATH', 'AQuA', 'ASDiv']:
        #     problems = [data.get('problem') for data in self.dataset]
        # elif self.dataset_name == 'ASDiv':
        #     conditions = [data.get('Body') for data in self.dataset]
        #     query = [data.get('Question') for data in self.dataset]
        #     problems = [conditions[idx] + ' ' + query[idx] for idx in range(len(conditions))]
        if self.dataset_name == 'amc23':
            problems = [data.get('question') for data in self.dataset]
        elif self.dataset_name == 'olympiad':
            problems = [data.get('question') for data in self.dataset]
        elif self.dataset_name == 'math500':
            problems = [data.get('problem') for data in self.dataset]
        elif self.dataset_name == 'minerva':
            problems = [data.get('question') for data in self.dataset]
        elif self.dataset_name == 'aime2024':
            problems = [data.get('Problem') for data in self.dataset]
        elif self.dataset_name == 'aime2025':
            problems = [data.get('problem') for data in self.dataset]
        else:
            raise KeyError()
            # problems = [data.get('original_question') for data in self.dataset]
        # if self.dataset_name == 'AQuA':
        #     options = self.get_options()
        #     problems = [f'{problems[idx]} \nAnswer Choices: {options[idx]}' for idx in range(len(problems))]
        self.problems = problems
        return problems

    def get_gold_answers(self):
        # gold_answers = [data.get('gold_answer') if self.dataset_name != 'ASDiv' else data.get('Answer') for data in self.dataset]
        if self.dataset_name == 'amc23':
            gold_answers = [data.get('answer') for data in self.dataset]
        elif self.dataset_name == 'math500':
            gold_answers = [data.get('answer') for data in self.dataset]
        elif self.dataset_name == 'olympiad':
            gold_answers = [data.get('final_answer')[0] for data in self.dataset]
        elif self.dataset_name == 'minerva':
            gold_answers = [data.get('answer') for data in self.dataset]
        elif self.dataset_name == 'aime2024':
            gold_answers = [data.get('Solution') for data in self.dataset]
        elif self.dataset_name == 'aime2025':
            gold_answers = [str(data.get('answer')) for data in self.dataset]
        else:
            raise KeyError()

        self.gold_answers = gold_answers
        return gold_answers

    # def get_gold_reasoning_paths(self):
    #     gold_reasoning_paths = [data.get('equation') if self.dataset_name != 'ASDiv' else data.get('Formula') for data in self.dataset]
    #     return gold_reasoning_paths

    # def get_options(self):
    #     options = [data.get('options') for data in self.dataset]
    #     options = ['(' + '('.join(i) for i in options]
    #     options = [i.replace('(', ' (').replace(')', ') ') for i in options]
    #     return options

    # def get_other_information(self):
    #     subject = [data.get('subject') for data in self.dataset]
    #     level = [data.get('level') for data in self.dataset]
    #     return subject, level

    # def print_dataset_info(self):
    #     print(f'[INFO] {self.dataset_name} - Path of dataset: {self.dataset_path}')
    #     print(f'[INFO] {self.dataset_name} - Number of math word problems: {len(self.dataset)}')

    # def print_avg_tokens(self, problems):
    #     print(f'[INFO] {self.dataset_name} - Average tokens in problems: {sum([num_tokens_from_string(problem) for problem in problems]) / (len(problems)):.2f}')