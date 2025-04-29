from phishing.url_extractor import extract_url_features
from urllib.parse import urlparse, parse_qs
import idna
import streamlit as st
from difflib import SequenceMatcher


def predictor(model, url: str , model_feature_names_file_path:str):
    # Initialize the URLFeatureExtractor to parse the features from the URL
    df = extract_url_features(url=url , model_feature_names_path=model_feature_names_file_path)

    # Define known whitelisted and blacklisted domains
    WHITELIST_DOMAINS = {
        "google.com", "accounts.google.com", "github.com", "chatgpt.com",
        "facebook.com", "twitter.com", "linkedin.com", "microsoft.com",
        "apple.com", "amazon.com", "youtube.com", "instagram.com", "yahoo.com",
        "reddit.com", "netflix.com", "wikipedia.org", "ebay.com", "dropbox.com",
        "whatsapp.com", "zoom.us", "icloud.com", "bing.com", "stackoverflow.com",
        "adobe.com", "paypal.com", "spotify.com", "paypal.me", "pinterest.com",
        "wordpress.com", "airbnb.com", "twitch.tv", "vimeo.com", "tumblr.com",
        "tripadvisor.com", "bbc.com", "bbc.co.uk", "nytimes.com", "cnn.com",
        "forbes.com", "wellsfargo.com", "chase.com", "bankofamerica.com", "usps.com",
        "bloomberg.com", "hulu.com", "etsy.com", "target.com", "shopify.com",
        "bestbuy.com", "costco.com", "nike.com", "adidas.com", "fandango.com",
        "expedia.com", "uber.com", "lyft.com", "grubhub.com", "doordash.com",
        "samsung.com", "lg.com", "merriam-webster.com", "theguardian.com",
        "theverge.com", "vox.com", "businessinsider.com", "theatlantic.com",
        "usatoday.com", "newyorker.com", "thehill.com", "politico.com",
        "washingtonpost.com", "latimes.com", "dell.com", "hp.com", "lenovo.com",
        "bose.com", "sonos.com", "apple.co.uk", "time.com", "nationalgeographic.com",
        "ted.com", "wikihow.com", "wikimedia.org", "who.int", "fda.gov",
        "cdc.gov", "mayoclinic.org", "webmd.com", "healthline.com", "goodhousekeeping.com",
        "reuters.com", "marketwatch.com", "businessweek.com"
    }

    BLACKLIST_DOMAINS = {
        "phishingsite.com", "malicious-example.com", "scam-alert.org", "fake-bank-login.com",
        "paypal-secure-login.com", "googIe.com", "appleid-login.com", "amazon-verify.net",
        "secure-icloudlogin.com", "login-paypaI.com", "freegiftcards.xyz", "account-verify.net",
        "bankofamerica-verification.com", "secure-dropboxlogin.com", "webmaiI.com", "netfix.com",
        "faceb00k.com", "tw1tter.com", "1inkedin.com", "rnicrosoft.com", "yaho0.com",
        "instagrarn.com", "b1ng.com", "airbnb-bonus.com", "secure-hulu.com", "update-whatsapp.com",
        "freemoneyclaim.com", "gov-verification.com", "unlock-youraccount.com", "free-netflixnow.com",
        "office365securelogin.com", "icloud-alert.com", "vaccine-check.com", "govrefunds.net",
        "youku-login.com", "freemembership.site", "urgent-banknotice.com", "steamgiftcard.xyz",
        "xlogin.com", "bit-c0inwallet.com", "deutsche-bankverify.com", "mytwitch.info",
        "recoverycenter.help", "microsoft-reset-password.com", "onedrive-update.com", "yourvirusalert.com",
        "applepayalert.com", "resetpin.info", "moneyclaim-alert.com", "malvertising-check.com",
        "suspicious-login.net", "upgrade-mailbox.com", "trustedsendercheck.com", "user-suspension.com",
        "zoom-invite-now.com", "whatsappweb.online", "chatgpt-signin.com", "amazon-alert-support.com",
        "dropbox-password-reset.com", "instagram-verify-id.com", "giftcard-check.xyz", "studentaid-login.net",
        "edu-verification.com", "secureappstoreupdate.com", "emailrecoverysite.com", "citibank-login-alert.com",
        "outlook365-securemail.com", "freedomainhoster.tk", "scampage-verify.com", "security-drop.com",
        "loginpage-reset.com", "surveywinnings.net", "identityrecovery.online", "webchatlogin.net",
        "ic0ud.com", "paypallogin-confirm.com", "visa-checkupdate.com", "mcafee-check.com",
        "bitly-login.com", "cuttly-access.com", "t.co-reset.com", "url-short-check.com",
        "verify-mytokens.com", "socialmedialogin.net", "vpn-safetyalert.com", "winfreebtc.com",
        "online-checkapp.com", "gov-id-update.com", "domainupdate-alert.com", "resetlogin-token.com",
        "xn--goog1e-qsa.com", "xn--faceb0ok-nf3e.com", "xn--paypa1-mta.com", "xn--micros0ft-8sd.com",
        "xn--1inkedin-hk3e.com", "xn--netf1ix-7za.com", "xn--ama2on-jf3e.com", "xn--twltter-pf3e.com" ,
        "amaz0n.com" ,"www.amaz0n.com" , "faceb00k.com" , "g00gle.com"
    }

    # Parse the URL to extract domain information
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    base_domain = ".".join(domain.split(".")[-2:])

    # Function to check domain similarity (for suspicious domains)
    def similar(a, b):
        return SequenceMatcher(None, a, b).ratio()

    # Function to check suspicious domains by similarity to whitelisted domains
    def check_suspicious_domain(domain):
        for known_domain in WHITELIST_DOMAINS:
            similarity = similar(domain, known_domain)
            if similarity > 0.8 and domain != known_domain:
                st.warning(f"⚠️ Domain '{domain}' is suspiciously similar to known domain '{known_domain}'")
                return True
        return False

     # 1. Check if domain is whitelisted
    if base_domain in WHITELIST_DOMAINS:
        st.success(f"✅ Whitelisted domain: {base_domain} : Safe.")
        y_pred = 0
        y_proba = [[1, 0]]
        return y_pred, y_proba
    
    # 2. Check for newly registered domain
    if df["time_domain_activation"][0] < 30:
        st.warning(f"⚠️ Domain '{domain}' is newly registered ({df['time_domain_activation'][0]} days old)")
        df["qty_redirects"] += 1

    # 3. Check for missing nameservers (suspicious)
    if "qty_nameservers" in df.columns and df["qty_nameservers"][0] == 0:
        st.warning("⚠️ No nameservers found — suspicious phishing attempt")
        df["qty_redirects"] += 1

    # 4. OAuth redirect detection
    query = parse_qs(parsed.query)
    if "oauth" in url.lower() and "redirect_uri" in query:
        redirect_target = query["redirect_uri"][0]
        redirect_domain = urlparse(redirect_target).netloc
        if not any(redirect_domain.endswith(whitelisted) for whitelisted in WHITELIST_DOMAINS):
            st.warning(f"⚠️ Suspicious OAuth redirect: {redirect_target}")
            df["qty_redirects"] += 1

    # 5. Homograph attack detection using IDNA
    try:
        decoded = idna.decode(domain)
        if decoded != domain:
            df["url_shortened"] = 1
            st.warning(f"⚠️ Homograph attack detected: {domain} → {decoded}")
    except idna.IDNAError:
        df["url_shortened"] = 1

    # 6. Suspicious TLD check
    SUSPICIOUS_TLDS = {"tk", "ml", "ga", "cf", "gq", "xyz", "top", "biz", "click"}
    tld = domain.split(".")[-1].lower()
    if tld in SUSPICIOUS_TLDS:
        st.warning(f"⚠️ Suspicious TLD: .{tld}")
        df["qty_redirects"] += 1

    # 7. Deep subdomain detection (common in phishing URLs)
    if domain.count(".") > 3:
        st.warning(f"⚠️ Deep subdomain in '{domain}'")
        df["qty_nameservers"] = 0

    # 8. Check if domain is blacklisted
    if base_domain in BLACKLIST_DOMAINS:
        st.warning(f"🚨 Blacklisted domain: {base_domain} — highly suspicious")
        y_pred = 1
        y_proba = [[0, 1]]
        return y_pred, y_proba

    # 9. Check domain similarity to known whitelisted domains
    if check_suspicious_domain(domain):
        df["qty_redirects"] += 1

   
   # 10. URL shortener check (common in phishing links)
    SHORTENERS = ["bit.ly", "tinyurl", "goo.gl", "t.co", "cutt.ly"]
    if any(short in url.lower() for short in SHORTENERS):
        st.warning(f"⚠️ URL uses shortener: {url}")
        df["url_shortened"] = 1

    # Final prediction based on extracted features
    y_pred = model.predict(df)[0]  # Predicted class (e.g., phishing or not)
    y_proba = model.predict_proba(df)  # Probability for each class

    return y_pred, y_proba
