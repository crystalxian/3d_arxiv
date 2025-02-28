import arxiv
import datetime
import pandas as pd
from tqdm import tqdm
import time

class ArxivScraper:
    def __init__(self, categories=None, keywords=None):
        """
        Initialize the arXiv scraper with categories and keywords
        
        Args:
            categories (list): List of arXiv categories (e.g., ['cs.CV', 'cs.GR', 'cs.LG'])
            keywords (list): List of keywords to filter papers (e.g., ['3D generation', 'NeRF'])
        """
        self.categories = categories or ['cs.CV', 'cs.GR', 'cs.LG']
        self.keywords = keywords or [
            '3D generation', 'NeRF', 'Neural Radiance Field', 
            'Diffusion Models', 'Gaussian Splatting', '3D Reconstruction',
            '3DGS', 'text-to-3D', 'image-to-3D', 'mesh generation',
            '3D representation', 'point cloud', 'implicit surface'
        ]
        
    def search_papers(self, days_back=1, max_results=100):
        """
        Search for papers published in the last N days
        
        Args:
            days_back (int): Number of days to look back
            max_results (int): Maximum number of results to return
            
        Returns:
            pandas.DataFrame: DataFrame containing paper information
        """
        date_start = datetime.datetime.now() - datetime.timedelta(days=days_back)
        date_start = date_start.strftime('%Y-%m-%dT00:00:00Z')
        
        # Create the search query
        categories_query = ' OR '.join([f'cat:{cat}' for cat in self.categories])
        date_query = f'submittedDate:[{date_start} TO now]'
        query = f'({categories_query}) AND ({date_query})'
        
        # Search arXiv
        client = arxiv.Client(page_size=100, delay_seconds=3)
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        
        # Process results
        papers = []
        for result in tqdm(client.results(search), desc="Fetching papers"):
            paper_info = {
                'id': result.entry_id.split('/')[-1],
                'title': result.title,
                'authors': ', '.join(author.name for author in result.authors),
                'abstract': result.summary,
                'categories': result.categories,
                'published': result.published,
                'updated': result.updated,
                'pdf_url': result.pdf_url,
                'entry_id': result.entry_id,
                'comment': result.comment if hasattr(result, 'comment') else None,
            }
            papers.append(paper_info)
            time.sleep(0.5)  # Be nice to the API
        
        # Convert to DataFrame
        df = pd.DataFrame(papers)
        
        # Filter by keywords
        if self.keywords:
            filtered_df = self._filter_by_keywords(df)
            print(f"Found {len(filtered_df)} papers after keyword filtering")
            return filtered_df
        
        return df
    
    def _filter_by_keywords(self, df):
        """Filter papers by keywords in title or abstract"""
        def contains_keywords(row):
            text = f"{row['title']} {row['abstract']}".lower()
            return any(keyword.lower() in text for keyword in self.keywords)
        
        return df[df.apply(contains_keywords, axis=1)]
    
    def save_papers(self, df, output_path='papers.csv'):
        """Save papers to CSV file"""
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} papers to {output_path}")

# Example usage
if __name__ == "__main__":
    scraper = ArxivScraper()
    papers_df = scraper.search_papers(days_back=2, max_results=200)
    scraper.save_papers(papers_df, output_path='3d_gen_papers_latest.csv')
