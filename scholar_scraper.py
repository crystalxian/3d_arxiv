import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
import time
import random
import datetime
from fake_useragent import UserAgent

class GoogleScholarScraper:
    def __init__(self, proxy=None):
        """
        Initialize the Google Scholar scraper
        
        Args:
            proxy (str): Optional proxy for making requests
        """
        self.base_url = "https://scholar.google.com/scholar"
        self.proxy = proxy
        self.ua = UserAgent()
        
    def search_papers(self, query, days_back=7, num_results=50):
        """
        Search for papers on Google Scholar
        
        Args:
            query (str): Search query
            days_back (int): Number of days to look back
            num_results (int): Maximum number of results to return
            
        Returns:
            pandas.DataFrame: DataFrame containing paper information
        """
        # Add date filter to query
        if days_back > 0:
            query += f" after:{self._get_date_str(days_back)}"
            
        papers = []
        results_per_page = 10
        num_pages = min(num_results // results_per_page + 1, 10)  # Google Scholar typically limits to ~10 pages
        
        for page in range(num_pages):
            if len(papers) >= num_results:
                break
                
            start_idx = page * results_per_page
            
            # Set headers with random user agent to avoid blocking
            headers = {
                'User-Agent': self.ua.random,
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Referer': 'https://scholar.google.com/'
            }
            
            params = {
                'q': query,
                'hl': 'en',
                'start': start_idx
            }
            
            try:
                if self.proxy:
                    proxies = {'http': self.proxy, 'https': self.proxy}
                    response = requests.get(
                        self.base_url, 
                        params=params, 
                        headers=headers, 
                        proxies=proxies,
                        timeout=30
                    )
                else:
                    response = requests.get(
                        self.base_url, 
                        params=params, 
                        headers=headers,
                        timeout=30
                    )
                
                if response.status_code != 200:
                    print(f"Failed to retrieve results: HTTP {response.status_code}")
                    break
                    
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract papers from current page
                article_elements = soup.select('div.gs_r.gs_or.gs_scl')
                
                for article in article_elements:
                    title_elem = article.select_one('h3.gs_rt')
                    if not title_elem:
                        continue
                        
                    # Clean title
                    title = title_elem.get_text().strip()
                    if title.startswith('[PDF]') or title.startswith('[HTML]'):
                        title = title.split('] ', 1)[1]
                        
                    # Get link
                    link_elem = title_elem.find('a')
                    link = link_elem['href'] if link_elem else ''
                    
                    # Get snippet (abstract)
                    snippet_elem = article.select_one('div.gs_rs')
                    snippet = snippet_elem.get_text().strip() if snippet_elem else ''
                    
                    # Get authors, publication, year
                    pub_info_elem = article.select_one('div.gs_a')
                    pub_info = pub_info_elem.get_text().strip() if pub_info_elem else ''
                    
                    # Parse publication info
                    authors = ''
                    venue = ''
                    year = ''
                    
                    if pub_info:
                        parts = pub_info.split(' - ')
                        authors = parts[0] if len(parts) > 0 else ''
                        
                        if len(parts) > 1:
                            venue_year = parts[1]
                            year_match = re.search(r', (\d{4})', venue_year)
                            if year_match:
                                year = year_match.group(1)
                                venue = venue_year.split(', ' + year)[0]
                    
                    # Get citation count
                    cit_elem = article.select_one('div.gs_fl a:nth-child(3)')
                    citations = 0
                    if cit_elem:
                        cit_text = cit_elem.get_text().strip()
                        cit_match = re.search(r'Cited by (\d+)', cit_text)
                        if cit_match:
                            citations = int(cit_match.group(1))
                    
                    paper_info = {
                        'id': f"scholar_{len(papers)}",
                        'title': title,
                        'authors': authors,
                        'abstract': snippet,
                        'link': link,
                        'venue': venue,
                        'year': year,
                        'citations': citations,
                        'source': 'Google Scholar',
                        'query': query
                    }
                    
                    papers.append(paper_info)
                    
                # Random delay to avoid detection
                delay = random.uniform(3, 8)
                time.sleep(delay)
                
            except Exception as e:
                print(f"Error while scraping Google Scholar: {e}")
                break
                
        return pd.DataFrame(papers)
    
    def _get_date_str(self, days_back):
        """Get date string for Google Scholar's 'after:' filter"""
        date = datetime.datetime.now() - datetime.timedelta(days=days_back)
        return date.strftime('%Y/%m/%d')
    
    def search_3d_papers(self, days_back=7, max_results=100):
        """
        Search for 3D-generation related papers
        
        Returns:
            pandas.DataFrame: DataFrame with papers
        """
        queries = [
            "3D generation",
            "NeRF Neural Radiance Fields",
            "Gaussian Splatting 3D",
            "3D Diffusion Models",
            "text-to-3D generation"
        ]
        
        all_papers = []
        
        for query in tqdm(queries, desc="Processing queries"):
            df = self.search_papers(query, days_back=days_back, num_results=max_results//len(queries))
            all_papers.append(df)
            
            # Be nice and wait between queries
            time.sleep(random.uniform(10, 15))
            
        # Combine results
        combined_df = pd.concat(all_papers, ignore_index=True)
        
        # Remove duplicates based on title
        deduplicated_df = combined_df.drop_duplicates(subset=['title'])
        
        return deduplicated_df
    
    def save_papers(self, df, output_path='scholar_papers.csv'):
        """Save papers to CSV file"""
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} papers to {output_path}")

# Example usage
if __name__ == "__main__":
    # Note: Using Google Scholar may require proxies or rotation strategies
    # to avoid being blocked. This is a basic implementation.
    scraper = GoogleScholarScraper()
    
    try:
        papers_df = scraper.search_3d_papers(days_back=7, max_results=100)
        scraper.save_papers(papers_df, output_path='3d_scholar_papers.csv')
    except Exception as e:
        print(f"Failed to scrape Google Scholar: {e}")
        print("Consider using a proxy or rotating IP addresses")
