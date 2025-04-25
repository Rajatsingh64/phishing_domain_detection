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
from  bs4 import BeautifulSoup
import ssl
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse
from ipwhois import IPWhois
import warnings
warnings.filterwarnings("ignore")

def get_asn_for_url(url):
    """
    Resolve the domain to an IP address and get the ASN (Autonomous System Number).

    Args:
        url (str): Domain or hostname.

    Returns:
        str: ASN if successful, else None.
    """
    try:
        ip_address = socket.gethostbyname(url)
        obj = IPWhois(ip_address)
        result = obj.lookup_rdap()
        return result.get('asn')
    except Exception as e:
        print(f"Error getting ASN for {url}: {e}")
        return 0


def time_response(url):
    """
    Measure the response time for a given URL in seconds.

    Args:
        url (str): Full URL to test.

    Returns:
        float: Response time in seconds, else -1 on failure.
    """
    try:
        response = requests.get(url)
        return response.elapsed.total_seconds()
    except requests.exceptions.RequestException as e:
        print(f"Request error for {url}: {e}")
        return -1


def domain_spf(domain):
    """
    Check if the domain has SPF (Sender Policy Framework) DNS record.

    Args:
        domain (str): Domain name.

    Returns:
        int: 1 if SPF found, 0 if not found, -1 on error.
    """
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
    """
    Count the number of IP addresses resolved for a domain.

    Args:
        domain (str): Domain name.

    Returns:
        int: Number of resolved IPs, or -1 on failure.
    """
    try:
        ips = socket.gethostbyname_ex(domain)
        return len(ips[2])
    except Exception as e:
        print(f"IP resolution error for {domain}: {e}")
        return 0


def qty_nameservers(domain):
    """
    Count the number of NS (Name Server) records for a domain.

    Args:
        domain (str): Domain name.

    Returns:
        int: Count of NS records or -1 on failure.
    """
    try:
        ns_records = dns.resolver.resolve(domain, "NS")
        return len(ns_records)
    except Exception as e:
        print(f"NS resolution error for {domain}: {e}")
        return 0


def qty_mx_servers(domain):
    """
    Count the number of MX (Mail Exchange) servers for a domain.

    Args:
        domain (str): Domain name.

    Returns:
        int: Count of MX servers or -1 on failure.
    """
    try:
        mx_records = dns.resolver.resolve(domain, "MX")
        return len(mx_records)
    except Exception as e:
        print(f"MX resolution error for {domain}: {e}")
        return 0


def tls_ssl_certificate(domain):
    """
    Check if the domain has a valid SSL certificate.

    Args:
        domain (str): Domain name.

    Returns:
        bool: True if valid certificate found, else False.
    """
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
    """
    Check if a URL is a shortened (redirect) link.

    Args:
        url (str): The full URL.

    Returns:
        bool: True if URL redirects, else False.
    """
    try:
        response = requests.head(url, allow_redirects=True)
        # Check if the status code is a redirection (3xx)
        if 300 <= response.status_code < 400:
            return True
        else:
            return False
    except requests.exceptions.RequestException:
        return False


def is_domain_indexed(domain):
    """
    Check if the domain is indexed by Google.

    Args:
        domain (str): Domain name.

    Returns:
        bool: True if indexed, else False.
    """
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
    """
    Check if the exact URL is indexed by Google.

    Args:
        url (str): The full URL.

    Returns:
        bool: True if indexed, else False.
    """
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
    """
    Calculate the number of days since domain registration.

    Args:
        domain (str): Domain name.

    Returns:
        int: Days since activation, or -1 on failure.
    """
    try:
        domain_info = whois.whois(domain)
        creation_date = domain_info.creation_date[0] if isinstance(domain_info.creation_date, list) else domain_info.creation_date
        return (datetime.now() - creation_date).days
    except Exception as e:
        print(f"Activation time error for {domain}: {e}")
        return -1


def time_domain_expiration(domain):
    """
    Calculate the number of days left before the domain expires.

    Args:
        domain (str): Domain name.

    Returns:
        int: Days until expiration, or -1 on failure.
    """
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
        return res.get("asn", -1)  # Return ASN or -1 if not found
    except Exception as e:
        print(f"ASN lookup failed: {e}")
        return -1

def get_ttl(domain):
    try:
        answer = dns.resolver.resolve(domain, 'A')
        # Return the TTL from the first answer
        return answer.rrset.ttl
    except Exception as e:
        print(f"DNS resolution failed: {e}")
        return -1  # or 0 or np.nan as fallback

def qty_ip_resolved(domain):
    try:
        ip_list = socket.gethostbyname_ex(domain)[-1]
        return len(ip_list)
    except Exception as e:
        return -1

def qty_redirects(url):
    try:
        response = requests.get(url, timeout=5)  # set timeout to avoid hanging
        return len(response.history)
    except Exception as e:
        return -1

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
            "qty_vowels_domain": sum(
                [self.domain.count(vowel) for vowel in "aeiouAEIOU"]
            ),
            "domain_length": len(self.domain),
            "domain_in_ip": 1 if self.domain.replace(".", "").isdigit() else 0,
            "server_client_domain": (
                1
                if "server" in self.domain.lower() or "client" in self.domain.lower()
                else 0
            ),
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
        return (
            directory_components
            if self.directory
            else {key: -1 for key in directory_components}
        )

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
            "tld_present_params": (
                1
                if self.parameters.endswith(".com")
                or self.parameters.endswith(".org")
                or self.parameters.endswith(".net")
                else 0
            ),
            "qty_params": len(self.parameters.split("&")),
        }
        return (
            parameters_components
            if self.parameters
            else {key: -1 for key in parameters_components}
        )

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
        for key, value in resolving_components.items():
               print(key, value)
        
        return resolving_components
    
    def get_external_services_components(self):
        return {}

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

    def get_all_components(self):
        return self.components

    def get_all_components_values(self):
        return list(self.components.values())

    def get_all_components_keys(self):
        return list(self.components.keys())
    
    def get_features_as_dataframe(self):
        data = self.get_all_components()
        a = [
            data["qty_slash_url"],
            data["time_domain_activation"],
            data["qty_dot_domain"],
            data["ttl_hostname"],
            data["asn_ip"],
            data["time_domain_expiration"],
            data["time_response"],
            data["qty_dot_url"],
            data["qty_vowels_domain"],
            data["qty_hyphen_url"],
            data["qty_hyphen_params"],
            data["qty_mx_servers"],
            data["qty_slash_params"],
            data["qty_percent_params"],
            data["qty_nameservers"],
            data["qty_redirects"],
            data["qty_equal_url"],
            data["qty_ip_resolved"],
            data["qty_underline_url"],
            data["tls_ssl_certificate"],
            data["domain_spf"],
            data["qty_tld_url"],
            data["qty_hyphen_domain"],
            data["url_shortened"],
            data["qty_percent_url"],
        ]
        columns = [
            "qty_slash_url", "time_domain_activation", "qty_dot_domain",
            "ttl_hostname", "asn_ip", "time_domain_expiration",
            "time_response", "qty_dot_url", "qty_vowels_domain",
            "qty_hyphen_url", "qty_hyphen_params", "qty_mx_servers",
            "qty_slash_params", "qty_percent_params", "qty_nameservers",
            "qty_redirects", "qty_equal_url", "qty_ip_resolved",
            "qty_underline_url", "tls_ssl_certificate", "domain_spf",
            "qty_tld_url", "qty_hyphen_domain", "url_shortened",
            "qty_percent_url"
        ]
        return pd.DataFrame([a], columns=columns)