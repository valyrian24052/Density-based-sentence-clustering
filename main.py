from text_summarizer import TextSummarizer
from dataset_evaluator import DatasetEvaluator
import gensim.downloader

def main():
    """
    Main function to initialize the summarizer and evaluator,
    and perform the evaluation on the datasets.
    """
    w2v_vectors = gensim.downloader.load('word2vec-google-news-300')
    summarizer = TextSummarizer(w2v_vectors)
    evaluator = DatasetEvaluator(summarizer)
    results = evaluator.evaluate()
    
    evaluator.print_results(results)

if __name__ == '__main__':
    main()
