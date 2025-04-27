import sys
import os
import logging


def error_message_detail(error, error_detail: sys):
    """
    Generate a detailed error message including the filename, line number, and error message.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occurred in Python script [{0}] at line number [{1}] with error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )
    return error_message


class PhishingException(Exception):
    """
    Custom exception class to handle errors in the application with detailed error messages.
    """
    def __init__(self, error_message, error_detail: sys):
        """
        Initialize the SrcException with a detailed error message.
        """
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

        
    def __str__(self):
        """
        Return the error message string when the exception is raised.
        """
        return self.error_message