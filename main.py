import sys

from phishing.exception import PhishingException
from phishing.pipeline.training_pipeline import initiate_training_pipeline


if __name__ == "__main__":
    try:
        initiate_training_pipeline()
    except Exception as exc:
        raise PhishingException(exc, sys) from exc
