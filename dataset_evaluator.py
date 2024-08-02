import tensorflow_datasets as tfds
from rouge import Rouge

class DatasetEvaluator:
    """
    A class to evaluate text summarization on multiple datasets.
    
    Attributes:
    summarizer : TextSummarizer
        An instance of TextSummarizer to perform text summarization.
    datasets : dict
        A dictionary containing the datasets to be evaluated.
    """

    def __init__(self, summarizer):
        """
        Initializes the evaluator with a summarizer and loads datasets.

        Parameters:
        summarizer : TextSummarizer
            An instance of TextSummarizer to perform text summarization.
        """
        self.summarizer = summarizer
        self.datasets = {
            'cnn_dailymail': tfds.load('cnn_dailymail', split='test', as_supervised=True, shuffle_files=False),
            'gigaword': tfds.load('gigaword', split='test', as_supervised=True, shuffle_files=False),
        }

    def evaluate(self):
        """
        Evaluates the summarizer on the datasets and returns the results.

        Returns:
        dict
            A dictionary containing the evaluation scores for each dataset.
        """
        results = {}

        for name, dataset in self.datasets.items():
            hyps = []
            refs = []

            for text, ref in dataset:
                text = text.numpy().decode('utf-8')
                hyps.append(self.summarizer.summarize(text))

                ref = ref.numpy().decode('utf-8')
                refs.append(ref)

            metric = Rouge()
            score = metric.get_scores(hyps, refs, avg=True)
            results[name] = score

        return results

    def print_results(self, results):
        """
        Prints the evaluation results.

        Parameters:
        results : dict
            A dictionary containing the evaluation scores for each dataset.
        """
        for name, score in results.items():
            print(name)
            print(f'Score: {score}')
