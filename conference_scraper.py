import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
from tqdm import tqdm
import json
import datetime
import re

class ConferenceScraper:
    def __init__(self):
        """Initialize the conference paper scraper"""
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Keywords for filtering
        self.keywords = [
            '3d', 'nerf', 'neural radiance field', 'diffusion', 'gaussian', 
            'reconstruction', 'rendering', 'text-to-3d', 'image-to-3d', 
            'mesh', 'point cloud', 'implicit surface'
        ]
        
    def scrape_openreview(self, conference="ICLR.cc/2024/Conference"):
        """
        Scrape papers from OpenReview
        
        Args:
            conference (str): Conference identifier
            
        Returns:
            pandas.DataFrame: DataFrame containing paper information
        """
        base_url = "https://api2.openreview.net/notes"
        params = {
            'invitation': f'{conference}/-/Submission',
            'details': 'replyCount,directReplies',
            'offset': 0,
            'limit': 1000
        }
        
        papers = []
        try:
            response = requests.get(base_url, params=params, headers=self.headers)
            data = response.json()
            
            if 'notes' not in data:
                print(f"No papers found for {conference}")
                return pd.DataFrame()
                
            for paper in tqdm(data['notes'], desc=f"Processing {conference} papers"):
                paper_info = {
                    'id': paper.get('id'),
                    'title': paper.get('content', {}).get('title'),
                    'authors': ', '.join(paper.get('content', {}).get('authors', [])),
                    'abstract': paper.get('content', {}).get('abstract', ''),
                    'keywords': ', '.join(paper.get('content', {}).get('keywords', [])),
                    'pdf_url': f"https://openreview.net/pdf?id={paper.get('id')}",
                    'conference': conference,
                    'source': 'OpenReview',
                    'published': datetime.datetime.fromtimestamp(paper.get('tmdate')/1000).strftime('%Y-%m-%d')
                }
                papers.append(paper_info)
                
        except Exception as e:
            print(f"Error scraping OpenReview: {e}")
            
        # Convert to DataFrame
        df = pd.DataFrame(papers)
        
        # Filter by keywords
        if not df.empty:
            filtered_df = self._filter_by_keywords(df)
            print(f"Found {len(filtered_df)} papers after keyword filtering")
            return filtered_df
            
        return df
    
    def scrape_cvpr(self, year=2024):
        """
        Scrape papers from CVPR
        
        Args:
            year (int): Year of the conference
            
        Returns:
            pandas.DataFrame: DataFrame containing paper information
        """
        papers = []
        
        # CVPR papers are usually on CVF Open Access after the conference
        # This is a placeholder - you may need to adjust based on current website structure
        url = f"https://openaccess.thecvf.com/CVPR{year}"
        
        try:
            response = requests.get(url, headers=self.headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Example extraction (adjust selectors based on actual structure)
            paper_items = soup.select('dt.ptitle')
            
            for item in tqdm(paper_items, desc=f"Processing CVPR {year} papers"):
                title = item.text.strip()
                
                # Find authors (adjust as needed)
                authors_elem = item.find_next('dd').find('div', {'class': 'authors'})
                authors = authors_elem.text.strip() if authors_elem else ''
                
                # Find abstract
                abstract_elem = item.find_next('dd').find('div', {'id': re.compile('abstract')})
                abstract = abstract_elem.text.strip() if abstract_elem else ''
                
                # Find PDF link
                pdf_link_elem = item.find_next('dd').find('a', href=re.compile(r'\.pdf$'))
                pdf_url = pdf_link_elem['href'] if pdf_link_elem else ''
                if pdf_url and not pdf_url.startswith('http'):
                    pdf_url = f"https://openaccess.thecvf.com/{pdf_url}"
                
                paper_info = {
                    'id': f"cvpr{year}_{len(papers)}",
                    'title': title,
                    'authors': authors,
                    'abstract': abstract,
                    'pdf_url': pdf_url,
                    'conference': f'CVPR {year}',
                    'source': 'CVF',
                    'published': f"{year}-06-01"  # Approximate date for CVPR
                }
                papers.append(paper_info)
                
        except Exception as e:
            print(f"Error scraping CVPR: {e}")
            
        # Convert to DataFrame
        df = pd.DataFrame(papers)
        
        # Filter by keywords
        if not df.empty:
            filtered_df = self._filter_by_keywords(df)
            print(f"Found {len(filtered_df)} papers after keyword filtering")
            return filtered_df
            
        return df
    
    def _filter_by_keywords(self, df):
        """Filter papers by keywords in title or abstract"""
        def contains_keywords(row):
            text = f"{row['title']} {row['abstract']} {row.get('keywords', '')}".lower()
            return any(keyword.lower() in text for keyword in self.keywords)
        
        return df[df.apply(contains_keywords, axis=1)]
    
    def save_papers(self, df, output_path='conference_papers.csv'):
        """Save papers to CSV file"""
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} papers to {output_path}")

# Example usage
if __name__ == "__main__":
    scraper = ConferenceScraper()
    
    # Scrape OpenReview papers
    openreview_df = scraper.scrape_openreview(conference="ICLR.cc/2024/Conference")
    
    # Scrape CVPR papers
    cvpr_df = scraper.scrape_cvpr(year=2024)
    
    # Combine results
    combined_df = pd.concat([openreview_df, cvpr_df], ignore_index=True)
    
    # Save results
    scraper.save_papers(combined_df, output_path='3d_conference_papers.csv')
