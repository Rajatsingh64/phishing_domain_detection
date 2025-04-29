import pandas as pd
import numpy as np
import re
import socket
import whois
from tld import get_tld
import requests
import tldextract
import dns.resolver
from dns import resolver
from datetime import datetime
from aslookup import get_as_data
from functools import lru_cache
from bs4 import BeautifulSoup
import ssl
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse
import pickle
from ipwhois import IPWhois
import warnings
warnings.filterwarnings("ignore")

def get_asn_for_url(url):
    try:
        ip_address = socket.gethostbyname(url)
        obj = IPWhois(ip_address)
        result = obj.lookup_rdap()
        return result.get('asn')
    except Exception as e:
        print(f"Error getting ASN for {url}: {e}")
        return 0


def time_response(url):
    try:
        response = requests.get(url)
        return response.elapsed.total_seconds()
    except requests.exceptions.RequestException as e:
        print(f"Request error for {url}: {e}")
        return -1


def domain_spf(domain):
    try:
        spf_records = dns.resolver.resolve(domain, "TXT")
        for record in spf_records:
            if "v=spf1" in record.to_text():
                return 1
        return 0
    except dns.resolver.NoAnswer:
        return 0
    except Exception as e:
        print(f"SPF check error for {domain}: {e}")
        return -1


def qty_ip_resolved(domain):
    try:
        ips = socket.gethostbyname_ex(domain)
        return len(ips[2])
    except Exception as e:
        print(f"IP resolution error for {domain}: {e}")
        return 0


def qty_nameservers(domain):
    try:
        ns_records = dns.resolver.resolve(domain, "NS")
        return len(ns_records)
    except Exception as e:
        print(f"NS resolution error for {domain}: {e}")
        return 0


def qty_mx_servers(domain):
    try:
        mx_records = dns.resolver.resolve(domain, "MX")
        return len(mx_records)
    except Exception as e:
        print(f"MX resolution error for {domain}: {e}")
        return 0


def tls_ssl_certificate(domain):
    try:
        context = ssl.create_default_context()
        with context.wrap_socket(socket.socket(), server_hostname=domain) as s:
            s.settimeout(5)
            s.connect((domain, 443))
            cert = s.getpeercert()
            return bool(cert and "subject" in cert and "issuer" in cert)
    except Exception as e:
        print(f"SSL error for {domain}: {e}")
        return False


def is_shortened_url(url):
    try:
        response = requests.head(url, allow_redirects=True)
        if 300 <= response.status_code < 400:
            return True
        else:
            return False
    except requests.exceptions.RequestException:
        return False


def is_domain_indexed(domain):
    try:
        search_url = f"https://www.google.com/search?q=site:{domain}"
        headers = {
            "User-Agent": "Mozilla/5.0"
        }
        response = requests.get(search_url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            for result in soup.find_all("cite"):
                if domain in result.text:
                    return True
        return False
    except Exception as e:
        print(f"Index check error for {domain}: {e}")
        return False


def is_url_indexed(url):
    try:
        search_url = f"https://www.google.com/search?q={url}"
        headers = {
            "User-Agent": "Mozilla/5.0"
        }
        response = requests.get(search_url, headers=headers)
        return url in response.text if response.status_code == 200 else False
    except Exception as e:
        print(f"URL index check error for {url}: {e}")
        return False


def time_domain_activation(domain):
    try:
        domain_info = whois.whois(domain)
        creation_date = domain_info.creation_date[0] if isinstance(domain_info.creation_date, list) else domain_info.creation_date
        return (datetime.now() - creation_date).days
    except Exception as e:
        print(f"Activation time error for {domain}: {e}")
        return -1


def time_domain_expiration(domain):
    try:
        domain_info = whois.whois(domain)
        expiration_date = domain_info.expiration_date[0] if isinstance(domain_info.expiration_date, list) else domain_info.expiration_date
        return (expiration_date - datetime.now()).days
    except Exception as e:
        print(f"Expiration time error for {domain}: {e}")
        return -1


def get_asn_ip(domain):
    try:
        ip = socket.gethostbyname(domain)
        obj = IPWhois(ip)
        res = obj.lookup_rdap()
        return res.get("asn", -1)
    except Exception as e:
        print(f"ASN lookup failed: {e}")
        return -1


def get_ttl(domain):
    try:
        answer = dns.resolver.resolve(domain, 'A')
        return answer.rrset.ttl
    except Exception as e:
        print(f"DNS resolution failed: {e}")
        return -1


def qty_redirects(url):
    try:
        response = requests.get(url, timeout=5)
        return len(response.history)
    except Exception as e:
        return -1

import pandas as pd

class URLFeatureExtractor:
    def __init__(self, url):
        self.url = url
        self.url_components = urlparse(url)
        self.domain = self.url_components.netloc
        self.directory = self.url_components.path
        self.file = self.url_components.path.split("/")[-1]
        self.parameters = self.url_components.query
        self.components = {}
        self.components.update(self.get_domain_components())
        self.components.update(self.get_directory_components())
        self.components.update(self.get_file_components())
        self.components.update(self.get_parameters_components())
        self.components.update(self.get_resolving_components())
        self.components.update(self.get_external_services_components())
        self.components.update(self.get_url_components())

    def get_domain_components(self):
        domain_components = {
            "qty_dot_domain": self.domain.count("."),
            "qty_hyphen_domain": self.domain.count("-"),
            "qty_underline_domain": self.domain.count("_"),
            "qty_slash_domain": self.domain.count("/"),
            "qty_questionmark_domain": self.domain.count("?"),
            "qty_equal_domain": self.domain.count("="),
            "qty_at_domain": self.domain.count("@"),
            "qty_and_domain": self.domain.count("&"),
            "qty_exclamation_domain": self.domain.count("!"),
            "qty_space_domain": self.domain.count(" "),
            "qty_tilde_domain": self.domain.count("~"),
            "qty_comma_domain": self.domain.count(","),
            "qty_plus_domain": self.domain.count("+"),
            "qty_asterisk_domain": self.domain.count("*"),
            "qty_hashtag_domain": self.domain.count("#"),
            "qty_dollar_domain": self.domain.count("$"),
            "qty_percent_domain": self.domain.count("%"),
            "qty_vowels_domain": sum([self.domain.count(vowel) for vowel in "aeiouAEIOU"]),
            "domain_length": len(self.domain),
            "domain_in_ip": 1 if self.domain.replace(".", "").isdigit() else 0,
            "server_client_domain": (1 if "server" in self.domain.lower() or "client" in self.domain.lower() else 0),
        }
        return domain_components

    def get_directory_components(self):
        directory_components = {
            "qty_dot_directory": self.directory.count("."),
            "qty_hyphen_directory": self.directory.count("-"),
            "qty_underline_directory": self.directory.count("_"),
            "qty_slash_directory": self.directory.count("/"),
            "qty_questionmark_directory": self.directory.count("?"),
            "qty_equal_directory": self.directory.count("="),
            "qty_at_directory": self.directory.count("@"),
            "qty_and_directory": self.directory.count("&"),
            "qty_exclamation_directory": self.directory.count("!"),
            "qty_space_directory": self.directory.count(" "),
            "qty_tilde_directory": self.directory.count("~"),
            "qty_comma_directory": self.directory.count(","),
            "qty_plus_directory": self.directory.count("+"),
            "qty_asterisk_directory": self.directory.count("*"),
            "qty_hashtag_directory": self.directory.count("#"),
            "qty_dollar_directory": self.directory.count("$"),
            "qty_percent_directory": self.directory.count("%"),
            "directory_length": len(self.directory),
        }
        return directory_components if self.directory else {key: -1 for key in directory_components}

    def get_file_components(self):
        file_components = {
            "qty_dot_file": self.file.count("."),
            "qty_hyphen_file": self.file.count("-"),
            "qty_underline_file": self.file.count("_"),
            "qty_slash_file": self.file.count("/"),
            "qty_questionmark_file": self.file.count("?"),
            "qty_equal_file": self.file.count("="),
            "qty_at_file": self.file.count("@"),
            "qty_and_file": self.file.count("&"),
            "qty_exclamation_file": self.file.count("!"),
            "qty_space_file": self.file.count(" "),
            "qty_tilde_file": self.file.count("~"),
            "qty_comma_file": self.file.count(","),
            "qty_plus_file": self.file.count("+"),
            "qty_asterisk_file": self.file.count("*"),
            "qty_hashtag_file": self.file.count("#"),
            "qty_dollar_file": self.file.count("$"),
            "qty_percent_file": self.file.count("%"),
            "file_length": len(self.file),
        }
        return file_components if self.file else {key: -1 for key in file_components}

    def get_parameters_components(self):
        parameters_components = {
            "qty_dot_params": self.parameters.count("."),
            "qty_hyphen_params": self.parameters.count("-"),
            "qty_underline_params": self.parameters.count("_"),
            "qty_slash_params": self.parameters.count("/"),
            "qty_questionmark_params": self.parameters.count("?"),
            "qty_equal_params": self.parameters.count("="),
            "qty_at_params": self.parameters.count("@"),
            "qty_and_params": self.parameters.count("&"),
            "qty_exclamation_params": self.parameters.count("!"),
            "qty_space_params": self.parameters.count(" "),
            "qty_tilde_params": self.parameters.count("~"),
            "qty_comma_params": self.parameters.count(","),
            "qty_plus_params": self.parameters.count("+"),
            "qty_asterisk_params": self.parameters.count("*"),
            "qty_hashtag_params": self.parameters.count("#"),
            "qty_dollar_params": self.parameters.count("$"),
            "qty_percent_params": self.parameters.count("%"),
            "params_length": len(self.parameters),
        }
        return parameters_components if self.parameters else {key: -1 for key in parameters_components}

    def get_resolving_components(self):
        try:
            ip = socket.gethostbyname(self.domain)
            resolving_components = {
                "time_response": time_response(self.url),
                "domain_spf": domain_spf(self.domain),
                "asn_ip": get_asn_ip(self.domain) ,
                "time_domain_activation": time_domain_activation(self.domain),
                "time_domain_expiration": time_domain_expiration(self.domain),
                "qty_ip_resolved": qty_ip_resolved(self.domain),
                "qty_nameservers": qty_nameservers(self.domain),
                "qty_mx_servers": qty_mx_servers(self.domain),
                "ttl_hostname": get_ttl(self.domain),
                "tls_ssl_certificate": 1 if tls_ssl_certificate(self.domain) else 0,
                "qty_redirects": qty_redirects(self.url),
                "url_google_index": 1 if is_url_indexed(self.url) else 0,
                "domain_google_index": 1 if is_domain_indexed(self.domain) else 0,
                "url_shortened": 1 if is_shortened_url(self.url) else 0,
            }

        except Exception as e:
            ip = 0  # In case the domain cannot be resolved
            print(f"{e}")
            resolving_components= {
            "time_response": -1,
            "domain_spf": -1,
            "asn_ip": -1,
            "time_domain_activation": -1,
            "time_domain_expiration": -1,
            "qty_ip_resolved": -1,
            "qty_nameservers": 0,
            "qty_mx_servers": 0,
            "ttl_hostname": -1,
            "tls_ssl_certificate": 0,
            "qty_redirects": -1,
            "url_google_index": 0,
            "domain_google_index": 0,
            "url_shortened": 0,
        }
        return resolving_components

    def get_external_services_components(self):
        return {
            "domain_spf": domain_spf(self.domain),
            "url_indexed": is_url_indexed(self.url),
            "domain_indexed": is_domain_indexed(self.domain),
        }

    def get_url_components(self):
        return {
            "qty_dot_url": self.url.count("."),
            "qty_hyphen_url": self.url.count("-"),
            "qty_underline_url": self.url.count("_"),
            "qty_slash_url": self.url.count("/"),
            "qty_questionmark_url": self.url.count("?"),
            "qty_equal_url": self.url.count("="),
            "qty_at_url": self.url.count("@"),
            "qty_and_url": self.url.count("&"),
            "qty_exclamation_url": self.url.count("!"),
            "qty_space_url": self.url.count(" "),
            "qty_tilde_url": self.url.count("~"),
            "qty_comma_url": self.url.count(","),
            "qty_plus_url": self.url.count("+"),
            "qty_asterisk_url": self.url.count("*"),
            "qty_hashtag_url": self.url.count("#"),
            "qty_dollar_url": self.url.count("$"),
            "qty_percent_url": self.url.count("%"),
            "qty_tld_url": len(self.url_components.netloc.split(".")[-1]),
            "length_url": len(self.url),
            "email_in_url": (
                1 if re.search(r"[a-zA-Z0-9]+@[a-zA-Z]+\.[a-zA-Z]+", self.url) else 0
            ),
        }

def extract_url_features(url: str, model_feature_names_path: str = None) -> pd.DataFrame:
 
    """
    Extracts features from the provided URL.
    
    Parameters:
        url (str): The URL from which features are extracted.
        model_feature_names_path (str): Path to the pickle file containing the feature names. Defaults to None.
    
    Returns:
        pd.DataFrame: A DataFrame with the extracted features.
    """

    # Initialize the URLFeatureExtractor with the provided URL
    feature_extractor = URLFeatureExtractor(url)
    
    # Load model_feature_names from pickle if path is provided
    model_feature_names = None
    if model_feature_names_path:
        try:
            with open(model_feature_names_path, 'rb') as f:
                model_feature_names = pickle.load(f)
        except Exception as e:
            raise Exception(f"Error loading model feature names from pickle file: {str(e)}")
    
    # Extract all features
    all_features = {
        **feature_extractor.get_domain_components(),
        **feature_extractor.get_directory_components(),
        **feature_extractor.get_file_components(),
        **feature_extractor.get_parameters_components(),
        **feature_extractor.get_resolving_components(),
        **feature_extractor.get_external_services_components(),
        **feature_extractor.get_url_components(),
    }
    
    # If model_feature_names is provided, extract only those features in the specified order
    if model_feature_names is not None and len(model_feature_names) > 0:
        ordered_features = {feature: all_features[feature] for feature in model_feature_names if feature in all_features}
        return pd.DataFrame([ordered_features])
    
    # Otherwise, return all features
    return pd.DataFrame([all_features])



