import os
import time
import schedule
import datetime
import pandas as pd
import logging
from tqdm import tqdm
import argparse

# Import our modules
from arxiv_scraper import ArxivScraper
from conference_scraper import ConferenceScraper
from scholar_scraper import GoogleScholarScraper
from database_manager import PaperDatabaseManager
from paper_classifier import PaperClassifier
from github_enricher import GitHubEnricher
from notification_system import NotificationSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("3dgen_agent.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PapersAgentController:
    def __init__(self, config=None):
        """
        Initialize the main controller with configuration
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = config or DEFAULT_CONFIG
        
        # Initialize components
        self.db_manager = PaperDatabaseManager(
            db_path=self.config.get('db_path', '3dgen_papers.db')
        )
        
        self.arxiv_scraper = ArxivScraper(
            categories=self.config.get('arxiv_categories', ['cs.CV', 'cs.GR', 'cs.LG']),
            keywords=self.config.get('keywords')
        )
        
        self.conference_scraper = ConferenceScraper()
        
        self.scholar_scraper = GoogleScholarScraper(
            proxy=self.config.get('proxy')
        )
        
        self.paper_classifier = PaperClassifier(
            api_key=self.config.get('openai_api_key'),
            model=self.config.get('model', 'deepseek-chat')
        )
        
        self.github_enricher = GitHubEnricher(
            api_key=self.config.get('github_token')
        )
        
        self.notification_system = NotificationSystem(
            email_config=self.config.get('email_config')
        )
        
        # Initialize tracking variables
        self.last_run = {
            'arxiv': None,
            'conference': None,
            'scholar': None,
            'enrichment': None,
            'notification': None
        }
    
    def run_daily_update(self):
        """Run the complete daily update pipeline"""
        logger.info("Starting daily update")
        
        try:
            # 1. Scrape papers from different sources
            self._scrape_papers()
            
            # 2. Classify and summarize papers
            self._classify_papers()
            
            # 3. Enrich papers with code and project links
            self._enrich_papers()
            
            # 4. Send notifications
            self._send_notifications()
            
            # Update last run time
            self.last_run['daily_update'] = datetime.datetime.now()
            
            logger.info("Daily update completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error in daily update: {e}", exc_info=True)
            return False
    
    def _scrape_papers(self):
        """Scrape papers from different sources"""
        logger.info("Scraping papers from sources")
        
        # 1. Scrape arXiv papers
        try:
            arxiv_papers = self.arxiv_scraper.search_papers(days_back=1, max_results=200)
            logger.info(f"Found {len(arxiv_papers)} papers from arXiv")
            
            if not arxiv_papers.empty:
                self.db_manager.import_papers(arxiv_papers, source='arxiv')
                
            self.last_run['arxiv'] = datetime.datetime.now()
        except Exception as e:
            logger.error(f"Error scraping arXiv: {e}", exc_info=True)
        
        # 2. Scrape conference papers (weekly)
        if self.last_run['conference'] is None or \
           (datetime.datetime.now() - self.last_run['conference']).days >= 7:
            try:
                # Get ICLR papers
                iclr_papers = self.conference_scraper.scrape_openreview(conference="ICLR.cc/2024/Conference")
                
                # Get CVPR papers
                cvpr_papers = self.conference_scraper.scrape_cvpr(year=2024)
                
                # Combine results
                conf_papers = pd.concat([iclr_papers, cvpr_papers], ignore_index=True)
                
                logger.info(f"Found {len(conf_papers)} papers from conferences")
                
                if not conf_papers.empty:
                    self.db_manager.import_papers(conf_papers, source='conference')
                    
                self.last_run['conference'] = datetime.datetime.now()
            except Exception as e:
                logger.error(f"Error scraping conferences: {e}", exc_info=True)
        
        # 3. Scrape Google Scholar (every 3 days to avoid blocking)
        if self.last_run['scholar'] is None or \
           (datetime.datetime.now() - self.last_run['scholar']).days >= 3:
            try:
                scholar_papers = self.scholar_scraper.search_3d_papers(days_back=3, max_results=50)
                logger.info(f"Found {len(scholar_papers)} papers from Google Scholar")
                
                if not scholar_papers.empty:
                    self.db_manager.import_papers(scholar_papers, source='scholar')
                    
                self.last_run['scholar'] = datetime.datetime.now()
            except Exception as e:
                logger.error(f"Error scraping Google Scholar: {e}", exc_info=True)
    
    def _classify_papers(self):
        """Classify and summarize recent papers"""
        logger.info("Classifying and summarizing papers")
        
        # Get unclassified papers from the last 7 days
        papers_to_classify = self._get_unclassified_papers(days=7)
        
        if papers_to_classify.empty:
            logger.info("No papers to classify")
            return
            
        logger.info(f"Classifying {len(papers_to_classify)} papers")
        
        # Process papers in batches to avoid API rate limits
        batch_size = 10
        for i in range(0, len(papers_to_classify), batch_size):
            batch = papers_to_classify.iloc[i:i+batch_size]
            
            try:
                self.paper_classifier.process_papers_batch(batch, self.db_manager)
                logger.info(f"Classified batch {i//batch_size + 1}/{(len(papers_to_classify)-1)//batch_size + 1}")
                
                # Sleep to avoid rate limits
                if i + batch_size < len(papers_to_classify):
                    time.sleep(2)
                    
            except Exception as e:
                logger.error(f"Error classifying paper batch: {e}", exc_info=True)
    
    def _enrich_papers(self):
        """Enrich papers with code and project links"""
        # Get papers without enrichment from the last 14 days
        if self.last_run['enrichment'] is None or \
           (datetime.datetime.now() - self.last_run['enrichment']).days >= 1:
            
            logger.info("Enriching papers with GitHub and project links")
            
            papers_to_enrich = self._get_unenriched_papers(days=14)
            
            if papers_to_enrich.empty:
                logger.info("No papers to enrich")
                return
                
            logger.info(f"Enriching {len(papers_to_enrich)} papers")
            
            try:
                self.github_enricher.enrich_papers(papers_to_enrich, self.db_manager)
                self.last_run['enrichment'] = datetime.datetime.now()
            except Exception as e:
                logger.error(f"Error enriching papers: {e}", exc_info=True)
    
    def _send_notifications(self):
        """Send notifications to subscribers"""
        logger.info("Preparing to send notifications")
        
        # Get all active subscriptions
        subscriptions = self.db_manager.get_active_subscriptions()
        
        if not subscriptions:
            logger.info("No active subscriptions to process")
            return
            
        logger.info(f"Processing {len(subscriptions)} subscriptions")
        
        for subscription in subscriptions:
            try:
                # Check if it's time to send this subscription based on frequency
                if not self._should_send_subscription(subscription):
                    continue
                    
                # Get papers for this subscription
                topic = subscription.get('topic', 'all')
                days = self._get_days_for_frequency(subscription.get('frequency', 'daily'))
                
                papers_df = self.db_manager.get_papers_for_subscription(topic, days=days)
                
                if papers_df.empty:
                    logger.info(f"No papers for subscription {subscription['id']}")
                    continue
                    
                # Send notification
                success = self.notification_system.send_subscription_notification(
                    subscription, papers_df, self.db_manager
                )
                
                if success:
                    logger.info(f"Sent notification for subscription {subscription['id']}")
                    
            except Exception as e:
                logger.error(f"Error processing subscription {subscription.get('id')}: {e}", exc_info=True)
                
        self.last_run['notification'] = datetime.datetime.now()
    
    def _get_unclassified_papers(self, days=7):
        """Get papers without classification"""
        # This would be implemented in the database manager
        # For now, we just get recent papers
        return self.db_manager.search_papers(days=days, limit=1000)
    
    def _get_unenriched_papers(self, days=14):
        """Get papers without code and project links"""
        # This would be implemented in the database manager
        # For now, we just get recent papers
        return self.db_manager.search_papers(days=days, has_code=False, limit=100)
    
    def _should_send_subscription(self, subscription):
        """Check if it's time to send this subscription"""
        frequency = subscription.get('frequency', 'daily')
        last_sent = subscription.get('last_sent')
        
        if not last_sent:
            return True
            
        if isinstance(last_sent, str):
            try:
                last_sent = datetime.datetime.fromisoformat(last_sent)
            except:
                return True
                
        now = datetime.datetime.now()
        
        if frequency == 'daily':
            # Check if it's been at least 20 hours since last sent
            return (now - last_sent).total_seconds() >= 20 * 3600
            
        elif frequency == 'weekly':
            # Check if it's been at least 6 days since last sent
            return (now - last_sent).days >= 6
            
        elif frequency == 'monthly':
            # Check if it's been at least 28 days since last sent
            return (now - last_sent).days >= 28
            
        return True
    
    def _get_days_for_frequency(self, frequency):
        """Get number of days to look back based on frequency"""
        if frequency == 'daily':
            return 1
        elif frequency == 'weekly':
            return 7
        elif frequency == 'monthly':
            return 30
        else:
            return 1
    
    def setup_scheduled_jobs(self):
        """Set up scheduled jobs"""
        # Run complete update at 8 AM daily
        schedule.every().day.at("08:00").do(self.run_daily_update)
        
        # Run scraper every 12 hours
        schedule.every(12).hours.do(self._scrape_papers)
        
        # Run enrichment process daily at 2 AM
        schedule.every().day.at("02:00").do(self._enrich_papers)
        
        # Run notification sender every 6 hours
        schedule.every(6).hours.do(self._send_notifications)
        
        logger.info("Scheduled jobs set up successfully")
    
    def run_scheduler(self):
        """Run the scheduler in a loop"""
        logger.info("Starting scheduler")
        
        while True:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except KeyboardInterrupt:
                logger.info("Scheduler stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in scheduler: {e}", exc_info=True)
                time.sleep(300)  # Wait 5 minutes after an error
    
    def generate_trends_report(self, days=30):
        """Generate trends report for the last N days"""
        logger.info(f"Generating trends report for the last {days} days")
        
        try:
            # Get papers from the last N days
            papers_df = self.db_manager.search_papers(days=days, limit=1000)
            
            if papers_df.empty:
                logger.info("No papers available for trends report")
                return {}
                
            # Analyze trends by category
            category_counts = {}
            if 'category' in papers_df.columns:
                category_counts = papers_df['category'].value_counts().to_dict()
                
            # Analyze trends by keywords
            keywords = self.db_manager.get_top_keywords(days=days, limit=20)
            
            # Analyze trends by code availability
            has_code_ratio = None
            if 'has_code' in papers_df.columns:
                has_code_count = papers_df['has_code'].sum()
                has_code_ratio = has_code_count / len(papers_df)
                
            # Build report
            report = {
                'total_papers': len(papers_df),
                'period_days': days,
                'generated_at': datetime.datetime.now().isoformat(),
                'category_distribution': category_counts,
                'top_keywords': keywords,
                'has_code_ratio': has_code_ratio
            }
            
            # Get the most cited papers
            if 'citations' in papers_df.columns:
                top_cited = papers_df.sort_values('citations', ascending=False).head(5)
                report['top_cited_papers'] = top_cited[['title', 'authors', 'citations']].to_dict('records')
                
            logger.info("Trends report generated successfully")
            return report
            
        except Exception as e:
            logger.error(f"Error generating trends report: {e}", exc_info=True)
            return {}
    
    def run_cli(self):
        """Run CLI interface for testing and manual operations"""
        parser = argparse.ArgumentParser(description='3D Generation Papers Agent')
        
        subparsers = parser.add_subparsers(dest='command', help='Command to run')
        
        # Command: run-update
        update_parser = subparsers.add_parser('run-update', help='Run the daily update')
        
        # Command: scrape
        scrape_parser = subparsers.add_parser('scrape', help='Scrape papers')
        scrape_parser.add_argument('--source', choices=['arxiv', 'conference', 'scholar', 'all'],
                                 default='all', help='Source to scrape')
        
        # Command: classify
        classify_parser = subparsers.add_parser('classify', help='Classify papers')
        classify_parser.add_argument('--days', type=int, default=7, help='Days to look back')
        
        # Command: enrich
        enrich_parser = subparsers.add_parser('enrich', help='Enrich papers')
        enrich_parser.add_argument('--days', type=int, default=14, help='Days to look back')
        
        # Command: notify
        notify_parser = subparsers.add_parser('notify', help='Send notifications')
        
        # Command: trends
        trends_parser = subparsers.add_parser('trends', help='Generate trends report')
        trends_parser.add_argument('--days', type=int, default=30, help='Days to look back')
        
        # Command: serve
        serve_parser = subparsers.add_parser('serve', help='Run scheduler')
        
        # Parse arguments
        args = parser.parse_args()
        
        if args.command == 'run-update':
            self.run_daily_update()
            
        elif args.command == 'scrape':
            if args.source == 'all' or args.source == 'arxiv':
                arxiv_papers = self.arxiv_scraper.search_papers(days_back=1, max_results=200)
                self.db_manager.import_papers(arxiv_papers, source='arxiv')
                
            if args.source == 'all' or args.source == 'conference':
                iclr_papers = self.conference_scraper.scrape_openreview(conference="ICLR.cc/2024/Conference")
                cvpr_papers = self.conference_scraper.scrape_cvpr(year=2024)
                conf_papers = pd.concat([iclr_papers, cvpr_papers], ignore_index=True)
                self.db_manager.import_papers(conf_papers, source='conference')
                
            if args.source == 'all' or args.source == 'scholar':
                scholar_papers = self.scholar_scraper.search_3d_papers(days_back=3, max_results=50)
                self.db_manager.import_papers(scholar_papers, source='scholar')
                
        elif args.command == 'classify':
            papers_to_classify = self._get_unclassified_papers(days=args.days)
            self.paper_classifier.process_papers_batch(papers_to_classify, self.db_manager)
            
        elif args.command == 'enrich':
            papers_to_enrich = self._get_unenriched_papers(days=args.days)
            self.github_enricher.enrich_papers(papers_to_enrich, self.db_manager)
            
        elif args.command == 'notify':
            self._send_notifications()
            
        elif args.command == 'trends':
            report = self.generate_trends_report(days=args.days)
            print(json.dumps(report, indent=2))
            
        elif args.command == 'serve':
            self.setup_scheduled_jobs()
            self.run_scheduler()
            
        else:
            parser.print_help()

    def process_papers(self, papers_df):
        """处理论文：分类和提取关键点"""
        processed_papers = []
        
        for _, paper in tqdm(papers_df.iterrows(), total=len(papers_df), desc="Processing papers"):
            try:
                # 分类论文
                try:
                    category, confidence = self.paper_classifier.classify_paper(paper)
                    paper['category'] = category
                    paper['confidence'] = confidence
                except Exception as e:
                    logger.error(f"Error classifying paper {paper.get('id', 'unknown')}: {str(e)}")
                    logger.debug(f"Paper data: {paper.to_dict()}")
                    paper['category'] = None
                    paper['confidence'] = 0.0

                # 提取关键点
                try:
                    key_points = self.paper_classifier.extract_key_points(paper)
                    paper['key_points'] = key_points
                except Exception as e:
                    logger.error(f"Error extracting key points for paper {paper.get('id', 'unknown')}: {str(e)}")
                    paper['key_points'] = []

                processed_papers.append(paper)
                
            except Exception as e:
                logger.error(f"Error processing paper {paper.get('id', 'unknown')}: {str(e)}")
                continue
            
        return pd.DataFrame(processed_papers)

# Example configuration
DEFAULT_CONFIG = {
    'db_path': '3dgen_papers.db',
    'arxiv_categories': ['cs.CV', 'cs.GR', 'cs.LG'],
    'keywords': [
        '3D generation', 'NeRF', 'Neural Radiance Field', 
        'Diffusion Models', 'Gaussian Splatting', '3D Reconstruction',
        '3DGS', 'text-to-3D', 'image-to-3D', 'mesh generation'
    ],
    'openai_api_key': os.environ.get('OPENAI_API_KEY'),
    'model': 'deepseek-chat',
    'github_token': os.environ.get('GITHUB_TOKEN'),
    'email_config': {
        'smtp_server': os.environ.get('SMTP_SERVER', 'smtp.gmail.com'),
        'smtp_port': int(os.environ.get('SMTP_PORT', 587)),
        'username': os.environ.get('EMAIL_USERNAME'),
        'password': os.environ.get('EMAIL_PASSWORD'),
        'sender': os.environ.get('EMAIL_SENDER')
    }
}

# Entry point
if __name__ == "__main__":
    import json
    
    # Load configuration from file if it exists
    config_path = 'config.json'
    config = DEFAULT_CONFIG
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config.update(json.load(f))
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
    
    # Initialize and run the agent
    agent = PapersAgentController(config)
    agent.run_cli()
