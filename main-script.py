#!/usr/bin/env python3
"""
3DGenPapersBot - Main entry point

This script initializes and runs the 3DGenPapersBot system.
It provides command line options to run different components separately,
or to start the complete system with the web server.
"""

import os
import logging
import argparse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def setup_directories():
    """Create necessary directories if they don't exist"""
    dirs = ['data', 'logs', 'static', 'static/charts']
    for dir_path in dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

# Create directories before configuring logging
setup_directories()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/3dgen_agent.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_server(debug=False, start_scheduler=False):
    """Run the Flask web server"""
    from flask_api import app, start_scheduler as start_flask_scheduler
    
    host = '0.0.0.0'
    port = int(os.environ.get('PORT', 5000))
    
    if start_scheduler:
        start_flask_scheduler()
        
    logger.info(f"Starting Flask server on {host}:{port} (debug={debug})")
    app.run(host=host, port=port, debug=debug)

def run_controller(command, **kwargs):
    """Run the controller with the specified command"""
    from main_controller import PapersAgentController, DEFAULT_CONFIG
    
    controller = PapersAgentController(DEFAULT_CONFIG)
    
    if command == 'serve':
        logger.info("Starting scheduler")
        controller.setup_scheduled_jobs()
        controller.run_scheduler()
    elif command == 'run-update':
        logger.info("Running daily update")
        controller.run_daily_update()
    elif command == 'scrape':
        source = kwargs.get('source', 'all')
        logger.info(f"Scraping papers from {source}")
        controller._scrape_papers()
    elif command == 'classify':
        days = kwargs.get('days', 7)
        logger.info(f"Classifying papers from the last {days} days")
        controller._classify_papers()
    elif command == 'enrich':
        days = kwargs.get('days', 14)
        logger.info(f"Enriching papers from the last {days} days")
        controller._enrich_papers()
    elif command == 'notify':
        logger.info("Sending notifications")
        controller._send_notifications()
    elif command == 'trends':
        days = kwargs.get('days', 30)
        logger.info(f"Generating trends report for the last {days} days")
        report = controller.generate_trends_report(days=days)
        print(report)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='3D Generation Papers Bot')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Server command
    server_parser = subparsers.add_parser('server', help='Run the web server')
    server_parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    server_parser.add_argument('--scheduler', action='store_true', help='Start scheduler')
    
    # Controller commands
    subparsers.add_parser('serve', help='Run scheduler')
    subparsers.add_parser('run-update', help='Run the daily update')
    
    scrape_parser = subparsers.add_parser('scrape', help='Scrape papers')
    scrape_parser.add_argument('--source', choices=['arxiv', 'conference', 'scholar', 'all'],
                             default='all', help='Source to scrape')
    
    classify_parser = subparsers.add_parser('classify', help='Classify papers')
    classify_parser.add_argument('--days', type=int, default=7, help='Days to look back')
    
    enrich_parser = subparsers.add_parser('enrich', help='Enrich papers')
    enrich_parser.add_argument('--days', type=int, default=14, help='Days to look back')
    
    subparsers.add_parser('notify', help='Send notifications')
    
    trends_parser = subparsers.add_parser('trends', help='Generate trends report')
    trends_parser.add_argument('--days', type=int, default=30, help='Days to look back')
    
    # Initialize command
    init_parser = subparsers.add_parser('init', help='Initialize the system')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create necessary directories
    setup_directories()
    
    # Execute command
    if args.command == 'server':
        run_server(debug=args.debug, start_scheduler=args.scheduler)
    elif args.command == 'init':
        from database_manager import PaperDatabaseManager
        db = PaperDatabaseManager()
        print("Database initialized successfully")
    elif args.command in ['serve', 'run-update', 'scrape', 'classify', 'enrich', 'notify', 'trends']:
        kwargs = {}
        if args.command == 'scrape' and hasattr(args, 'source'):
            kwargs['source'] = args.source
        if args.command in ['classify', 'enrich', 'trends'] and hasattr(args, 'days'):
            kwargs['days'] = args.days
        run_controller(args.command, **kwargs)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
