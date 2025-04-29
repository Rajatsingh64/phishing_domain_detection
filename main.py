from phishing.exception import PhishingException
from phishing.pipeline.training_pipeline import initiate_training_pipeline
import os, sys
import pandas as pd

if __name__ == "__main__":
    try:
        initiate_training_pipeline()

    except Exception as e:
        raise PhishingException(e, sys)
