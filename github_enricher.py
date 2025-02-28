import re
import requests
import time
import pandas as pd
from tqdm import tqdm
import os
import asyncio
import logging

logger = logging.getLogger(__name__)

class GitHubEnricher:
    def __init__(self, api_key=None):
        """
        Initialize the GitHub link detector and enricher
        
        Args:
            api_key (str): GitHub API token
        """
        self.api_key = api_key or os.getenv('GITHUB_TOKEN')
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({
                'Authorization': f'token {self.api_key}',
                'Accept': 'application/vnd.github.v3+json'
            })
    
    def extract_repo_info(self, url):
        """从URL中提取仓库信息"""
        try:
            # 支持多种URL格式
            patterns = [
                r'github\.com/([^/]+)/([^/]+)',  # 标准GitHub URL
                r'github\.io/([^/]+)/([^/]+)',   # GitHub Pages
                r'raw\.githubusercontent\.com/([^/]+)/([^/]+)'  # Raw content
            ]
            
            for pattern in patterns:
                match = re.search(pattern, url)
                if match:
                    return match.group(1), match.group(2)
            
            return None, None
            
        except Exception as e:
            logger.error(f"Error extracting repo info from URL {url}: {e}")
            return None, None

    async def enrich_paper(self, paper):
        """使用GitHub API丰富论文信息"""
        if not paper.get('code_url'):
            return paper
            
        owner, repo = self.extract_repo_info(paper['code_url'])
        if not owner or not repo:
            return paper
            
        try:
            # 清理repo名称（移除.git后缀等）
            repo = repo.replace('.git', '').strip()
            
            # 构建API URL
            api_url = f'https://api.github.com/repos/{owner}/{repo}'
            
            # 添加错误处理和重试逻辑
            for attempt in range(3):
                try:
                    response = self.session.get(api_url, timeout=10)
                    response.raise_for_status()
                    repo_data = response.json()
                    
                    # 更新论文信息
                    paper['github_stars'] = repo_data.get('stargazers_count', 0)
                    paper['github_forks'] = repo_data.get('forks_count', 0)
                    paper['last_updated'] = repo_data.get('updated_at')
                    paper['social_score'] = paper['github_stars'] * 2 + paper['github_forks']
                    
                    break
                except requests.exceptions.RequestException as e:
                    if attempt == 2:  # 最后一次尝试
                        logger.error(f"Error fetching repo data: {e.response.status_code} {e.response.text}")
                        break
                    await asyncio.sleep(1 * (attempt + 1))
                    
        except Exception as e:
            logger.error(f"Error enriching paper with GitHub data: {e}")
            
        return paper
    
    def extract_github_links(self, text):
        """
        Extract GitHub repository links from text
        
        Args:
            text (str): Text to search for GitHub links
            
        Returns:
            list: List of GitHub repository URLs
        """
        if not text:
            return []
            
        # GitHub repository regex patterns
        patterns = [
            r'https?://github\.com/[a-zA-Z0-9-]+/[a-zA-Z0-9._-]+',
            r'github\.com/[a-zA-Z0-9-]+/[a-zA-Z0-9._-]+'
        ]
        
        github_links = []
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                # Ensure URL starts with http protocol
                if not match.startswith('http'):
                    match = 'https://' + match
                    
                # Remove any trailing content after the repo name
                match = re.sub(r'(https?://github\.com/[a-zA-Z0-9-]+/[a-zA-Z0-9._-]+).*', r'\1', match)
                
                github_links.append(match)
                
        return list(set(github_links))  # Remove duplicates
    
    def get_repo_metadata(self, repo_url):
        """
        Get metadata for a GitHub repository
        
        Args:
            repo_url (str): GitHub repository URL
            
        Returns:
            dict: Repository metadata
        """
        # Extract owner and repo name from URL
        pattern = r'github\.com/([a-zA-Z0-9-]+)/([a-zA-Z0-9._-]+)'
        match = re.search(pattern, repo_url)
        
        if not match:
            return {}
            
        owner, repo = match.groups()
        
        # If no GitHub token, return basic info
        if not self.api_key:
            return {
                'url': repo_url,
                'owner': owner,
                'name': repo,
                'stars': None,
                'forks': None,
                'updated_at': None
            }
            
        # Query GitHub API
        api_url = f"https://api.github.com/repos/{owner}/{repo}"
        
        try:
            response = requests.get(api_url, headers=self.session.headers)
            
            if response.status_code != 200:
                print(f"Error fetching repo data: {response.status_code} {response.text}")
                return {
                    'url': repo_url,
                    'owner': owner,
                    'name': repo
                }
                
            data = response.json()
            
            return {
                'url': repo_url,
                'owner': owner,
                'name': repo,
                'description': data.get('description'),
                'stars': data.get('stargazers_count'),
                'forks': data.get('forks_count'),
                'language': data.get('language'),
                'created_at': data.get('created_at'),
                'updated_at': data.get('updated_at'),
                'topics': data.get('topics', [])
            }
            
        except Exception as e:
            print(f"Error fetching repo metadata: {e}")
            return {
                'url': repo_url,
                'owner': owner,
                'name': repo
            }
    
    def find_demo_and_project_links(self, text):
        """
        Find demo and project page links in text
        
        Args:
            text (str): Text to search for links
            
        Returns:
            dict: Dictionary with demo and project links
        """
        if not text:
            return {'project_url': None, 'demo_url': None}
            
        links = {}
        
        # Project website patterns (common in research papers)
        project_patterns = [
            r'https?://[a-zA-Z0-9-]+\.[a-zA-Z0-9-]+\.[a-zA-Z0-9-]+/~[a-zA-Z0-9-]+/projects/[a-zA-Z0-9-/]+',
            r'https?://[a-zA-Z0-9-]+\.[a-zA-Z0-9-]+\.[a-zA-Z0-9-]+/projects/[a-zA-Z0-9-/]+',
            r'https?://[a-zA-Z0-9-]+\.[a-zA-Z0-9-]+\.[a-zA-Z0-9-]+/research/[a-zA-Z0-9-/]+',
            r'https?://[a-zA-Z0-9-]+\.github\.io/[a-zA-Z0-9-/]+'
        ]
        
        # Find project URL
        for pattern in project_patterns:
            matches = re.findall(pattern, text)
            if matches:
                links['project_url'] = matches[0]
                break
                
        # Find demo video (YouTube, Vimeo)
        video_pattern = r'https?://(www\.)?(youtube\.com|youtu\.be|vimeo\.com)/[a-zA-Z0-9-/]+'
        video_matches = re.findall(video_pattern, text)
        
        if video_matches:
            links['demo_url'] = ''.join(video_matches[0])
            
        return links
    
    def enrich_papers(self, papers_df, db_manager=None):
        """
        Enrich papers with GitHub, project, and demo links
        
        Args:
            papers_df (pandas.DataFrame): DataFrame containing papers
            db_manager: Optional database manager to update papers
            
        Returns:
            pandas.DataFrame: Enriched papers DataFrame
        """
        enriched_papers = []
        
        for _, paper in tqdm(papers_df.iterrows(), total=len(papers_df), desc="Enriching papers"):
            paper_dict = paper.to_dict()
            
            # Combine title and abstract for searching
            text = f"{paper_dict.get('title', '')} {paper_dict.get('abstract', '')}"
            
            # Find GitHub links
            github_links = self.extract_github_links(text)
            
            if github_links:
                paper_dict['code_url'] = github_links[0]  # Use first link found
                paper_dict['has_code'] = True
                
                # Get repo metadata
                repo_metadata = self.get_repo_metadata(github_links[0])
                paper_dict['repo_metadata'] = repo_metadata
                
            # Find project and demo links
            links = self.find_demo_and_project_links(text)
            
            if links.get('project_url'):
                paper_dict['project_url'] = links['project_url']
                
            if links.get('demo_url'):
                paper_dict['video_url'] = links['demo_url']
                
            enriched_papers.append(paper_dict)
            
            # Update database if provided
            if db_manager and paper_dict.get('id'):
                update_data = {
                    'code_url': paper_dict.get('code_url'),
                    'project_url': paper_dict.get('project_url'),
                    'video_url': paper_dict.get('video_url')
                }
                
                db_manager.update_paper_metadata(paper_dict['id'], update_data)
                
            # Rate limit for GitHub API
            if self.api_key and github_links:
                time.sleep(0.5)
                
        return pd.DataFrame(enriched_papers)

# Example usage
if __name__ == "__main__":
    from database_manager import PaperDatabaseManager
    
    # Initialize enricher
    enricher = GitHubEnricher()
    
    # Initialize database manager
    db_manager = PaperDatabaseManager()
    
    # Get papers that may need enrichment
    papers_df = db_manager.search_papers(days=7)
    
    # Enrich papers
    enriched_df = enricher.enrich_papers(papers_df, db_manager)
    
    # Print enrichment stats
    has_code = enriched_df['has_code'].sum() if 'has_code' in enriched_df else 0
    has_project = enriched_df['project_url'].notna().sum()
    has_video = enriched_df['video_url'].notna().sum()
    
    print(f"Enriched {len(enriched_df)} papers:")
    print(f" - {has_code} papers with code")
    print(f" - {has_project} papers with project pages")
    print(f" - {has_video} papers with demo videos")
