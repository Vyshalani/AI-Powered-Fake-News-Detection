# retriever.py
from bs4 import BeautifulSoup
from urllib.parse import quote
import time 
import undetected_chromedriver as uc 
from selenium.webdriver.chrome.options import Options

# Configuration Constants
LOAD_TIMEOUT = 45  
ELEMENT_WAIT_TIME = 15 

TRUSTED_SITES = {
    # The Namibian 
    "The Namibian": ("https://www.namibian.com.na/?s=", 
                     "li.search-result a, article a, h2 a"), 
                     
    # Republikein 
    "Republikein": ("https://www.republikein.com.na/search?query=", 
                    ".article-title a, article a"),
                    
    # Kosmos 94.1
    "Kosmos 94.1": ("https://kosmos.com.na/?s=", 
                    "h3.entry-title a, h2.entry-title a"), 
                    
    # Namibian Sun  
    "Namibian Sun": ("https://www.namibiansun.com/search?query=", 
                     "div.article-post a, div.article-box a, li.search-result a"),
}

# Core Web Scraping Function 
def fetch_from_site(site_name, base_url, selector, query, driver, num_results=5):
    """
    Fetches article headlines and links.
    """
    url = base_url + quote(query)
    results = []

    print(f"[INFO] Searching {site_name} (using undetected-chromedriver) at: {url}")
    
    try:
        driver.get(url)
        time.sleep(5) 

        soup = BeautifulSoup(driver.page_source, "html.parser")
        base_domain = base_url.split('/')[2]

        # Iterate and Collect Results
        for a_tag in soup.select(selector):
            title = a_tag.get_text(strip=True)
            link = a_tag.get("href")

            if link and not link.startswith("http"):
                link = f"https://{base_domain}{link}"
            
            # Filter out short or non-article links, including boilerplate links
            if title and link and len(title) > 10 and title.lower() not in ['read more', 'continue reading', 'more info', 'more top stories']:
                
                formatted_result = f"{title} ({link})"
                
                if formatted_result not in results:
                    results.append(formatted_result)
                
            if len(results) >= num_results:
                break

        print(f"[INFO] Successfully retrieved {len(results)} results from {site_name}.")
        return results

    except Exception as e:
        print(f"[ERROR] {site_name} fetch failed: {e}")
        return []


# Main Evidence Retriever 
def fetch_evidence(claim_text, num_results=5):
    # Extract first 5-7 words as search query
    keywords = ' '.join(claim_text.split()[:7])
    print(f"🔍 Searching with keywords: '{keywords}'")
    
    evidence = []
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    print("[INFO] Starting Undetected-Chromedriver...")
    driver = None
    try:
        driver = uc.Chrome(options=chrome_options, headless=True)
        driver.set_page_load_timeout(LOAD_TIMEOUT)
        print("✅ ChromeDriver started successfully!")
        
    except Exception as e:
        print(f"[CRITICAL ERROR] Failed to start ChromeDriver: {e}")
        return []

    WORKING_SITES = {
        "The Namibian": TRUSTED_SITES["The Namibian"],
        "Republikein": TRUSTED_SITES["Republikein"],
        "Kosmos 94.1": TRUSTED_SITES["Kosmos 94.1"],
    }
    
    for site_name, (url, selector) in WORKING_SITES.items():
        print(f"🔍 Searching {site_name} for: '{keywords}'")
        snippets = fetch_from_site(site_name, url, selector, keywords, driver, num_results) 
        evidence.extend(snippets)
        
    if driver:
        driver.quit()
        print("✅ ChromeDriver closed successfully")

    return evidence

# Example usage 
if __name__ == "__main__":
    print("--- Web Scraping Retriever Test (undetected-chromedriver) ---")
    
    query = "education" 
    
    results = fetch_evidence(query, num_results=5)

    if results:
        print("\n=== Retrieved Headlines ===")
        for idx, headline in enumerate(results, start=1):
            print(f"{idx}. {headline}")
    else:
        print(f"\nNo headlines found for your query: '{query}'.")