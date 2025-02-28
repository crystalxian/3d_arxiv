# arxiv_collector.py
import arxiv
import datetime
import logging
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArxivCollector:
    """arXiv论文收集器"""
    
    def __init__(self, categories: List[str], keywords: List[str], max_results: int = 100):
        """
        初始化arXiv收集器
        
        Args:
            categories: arXiv分类如 ['cs.CV', 'cs.GR', 'cs.LG']
            keywords: 关键词过滤列表如 ['3D generation', 'NeRF']
            max_results: 每次查询最大结果数
        """
        self.categories = categories
        self.keywords = keywords
        self.max_results = max_results
        
    def get_recent_papers(self, days_back: int = 1) -> List[Dict[str, Any]]:
        """获取最近n天的论文"""
        since_date = datetime.datetime.now() - datetime.timedelta(days=days_back)
        
        # 构建查询
        query = ' OR '.join([f'cat:{category}' for category in self.categories])
        
        # 创建搜索客户端
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=self.max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        
        papers = []
        for result in client.results(search):
            # 检查发布日期
            if result.published > since_date:
                # 检查是否包含关键词
                if self._matches_keywords(result):
                    paper_info = {
                        'id': result.entry_id,
                        'title': result.title,
                        'authors': [author.name for author in result.authors],
                        'abstract': result.summary,
                        'pdf_url': result.pdf_url,
                        'published': result.published,
                        'categories': result.categories,
                        'source': 'arxiv',
                        'code_url': self._extract_code_url(result.summary)
                    }
                    papers.append(paper_info)
                    logger.info(f"Found paper: {result.title}")
            
        logger.info(f"Collected {len(papers)} papers from arXiv")
        return papers
    
    def _matches_keywords(self, paper) -> bool:
        """检查论文是否包含关键词"""
        text = (paper.title + ' ' + paper.summary).lower()
        return any(keyword.lower() in text for keyword in self.keywords)
    
    def _extract_code_url(self, text: str) -> str:
        """尝试从摘要中提取代码链接"""
        # 简单实现，可以用正则表达式优化
        if 'github.com' in text:
            start = text.find('github.com')
            end = text.find(' ', start)
            if end == -1:
                end = len(text)
            return 'https://' + text[start:end]
        return ""




