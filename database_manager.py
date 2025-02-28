import pandas as pd
import sqlite3
import json
import datetime
import os
from tqdm import tqdm
import logging

# Configure logging
logger = logging.getLogger(__name__)

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        return super().default(obj)

class PaperDatabaseManager:
    def __init__(self, db_path='3dgen_papers.db'):
        """
        Initialize the paper database manager
        
        Args:
            db_path (str): Path to SQLite database file
        """
        self.db_path = db_path
        self.setup_database()
        
    def setup_database(self):
        """Create the database schema if it doesn't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create papers table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS papers (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            authors TEXT,
            abstract TEXT,
            categories TEXT,
            published DATE,
            updated DATE,
            pdf_url TEXT,
            entry_id TEXT,
            source TEXT,
            venue TEXT,
            code_url TEXT,
            project_url TEXT,
            video_url TEXT,
            citations INTEGER DEFAULT 0,
            social_score INTEGER DEFAULT 0,
            has_code BOOLEAN DEFAULT 0,
            date_added TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata TEXT,
            summary TEXT
        )
        ''')
        
        # Create keywords table for faster searching
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS paper_keywords (
            paper_id TEXT,
            keyword TEXT,
            FOREIGN KEY (paper_id) REFERENCES papers(id),
            PRIMARY KEY (paper_id, keyword)
        )
        ''')
        
        # Create categories table for classification
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS paper_categories (
            paper_id TEXT,
            category TEXT,
            confidence REAL,
            FOREIGN KEY (paper_id) REFERENCES papers(id),
            PRIMARY KEY (paper_id, category)
        )
        ''')
        
        # Create user subscriptions table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_subscriptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            topic TEXT,
            frequency TEXT DEFAULT 'daily',
            format TEXT DEFAULT 'email',
            email TEXT,
            webhook_url TEXT,
            wechat_id TEXT,
            last_sent TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            active BOOLEAN DEFAULT 1
        )
        ''')
        
        # Create key points table for paper insights
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS paper_key_points (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_id TEXT,
            point TEXT,
            FOREIGN KEY (paper_id) REFERENCES papers(id)
        )
        ''')
        
        # Create trend analysis table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS trend_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_date DATE,
            period_days INTEGER,
            total_papers INTEGER,
            category_distribution TEXT,
            top_keywords TEXT,
            has_code_ratio REAL,
            report_data TEXT
        )
        ''')
        
        # Create index for faster searching
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_papers_published ON papers(published)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_papers_title ON papers(title)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_paper_keywords ON paper_keywords(keyword)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_paper_categories ON paper_categories(category)')
        
        conn.commit()
        conn.close()
        
        logger.info(f"Database initialized at {self.db_path}")
        
    def import_papers(self, dataframe, source):
        """
        Import papers from a dataframe into the database
        
        Args:
            dataframe (pandas.DataFrame): DataFrame containing paper information
            source (str): Source of the papers (e.g., 'arxiv', 'openreview')
            
        Returns:
            int: Number of papers added
        """
        if dataframe.empty:
            return 0
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get existing paper IDs to avoid duplicates
        cursor.execute('SELECT id FROM papers')
        existing_ids = set(row[0] for row in cursor.fetchall())
        
        papers_added = 0
        
        for _, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc=f"Importing {source} papers"):
            paper_id = row.get('id', '')
            
            # Skip if paper already exists
            if paper_id in existing_ids:
                continue
            
            # Convert lists and timestamps to strings before storing
            row_dict = row.to_dict()
            for key, value in row_dict.items():
                if isinstance(value, list):
                    row_dict[key] = ', '.join(map(str, value))
                elif isinstance(value, (pd.Timestamp, datetime.datetime)):
                    row_dict[key] = value.strftime('%Y-%m-%d %H:%M:%S')
                
            # Prepare data for insertion
            paper_data = {
                'id': paper_id,
                'title': row_dict.get('title', ''),
                'authors': row_dict.get('authors', ''),
                'abstract': row_dict.get('abstract', ''),
                'categories': row_dict.get('categories', ''),
                'published': row_dict.get('published', ''),
                'updated': row_dict.get('updated', ''),
                'pdf_url': row_dict.get('pdf_url', '') or row_dict.get('link', ''),
                'entry_id': row_dict.get('entry_id', ''),
                'source': source,
                'venue': row_dict.get('venue', '') or row_dict.get('conference', ''),
                'code_url': row_dict.get('code_url', ''),
                'project_url': row_dict.get('project_url', ''),
                'video_url': row_dict.get('video_url', ''),
                'citations': row_dict.get('citations', 0),
                'social_score': 0,
                'has_code': 1 if row_dict.get('code_url') else 0,
                'date_added': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'metadata': json.dumps(row_dict, cls=DateTimeEncoder),
                'summary': row_dict.get('summary', '')
            }
            
            # Insert paper
            placeholders = ', '.join(['?'] * len(paper_data))
            columns = ', '.join(paper_data.keys())
            sql = f'INSERT INTO papers ({columns}) VALUES ({placeholders})'
            
            try:
                cursor.execute(sql, list(paper_data.values()))
                papers_added += 1
                
                # Extract and store keywords
                keywords = self._extract_keywords(row)
                for keyword in keywords:
                    cursor.execute(
                        'INSERT OR IGNORE INTO paper_keywords (paper_id, keyword) VALUES (?, ?)',
                        (paper_id, keyword)
                    )
                    
                # If the paper has a category and confidence, add it
                if 'category' in row and row['category']:
                    confidence = row.get('confidence', 1.0)
                    cursor.execute(
                        'INSERT OR IGNORE INTO paper_categories (paper_id, category, confidence) VALUES (?, ?, ?)',
                        (paper_id, row['category'], confidence)
                    )
                    
                # If the paper has key points, add them
                if 'key_points' in row and isinstance(row['key_points'], list):
                    for point in row['key_points']:
                        cursor.execute(
                            'INSERT INTO paper_key_points (paper_id, point) VALUES (?, ?)',
                            (paper_id, point)
                        )
                    
            except sqlite3.IntegrityError as e:
                logger.warning(f"Integrity error when inserting paper {paper_id}: {e}")
                
        conn.commit()
        conn.close()
        
        logger.info(f"Added {papers_added} new papers from {source}")
        return papers_added
        
    def _extract_keywords(self, paper_row):
        """Extract keywords from paper data"""
        keywords = set()
        
        # Add explicit keywords if available
        if 'keywords' in paper_row and isinstance(paper_row['keywords'], str):
            keywords.update([k.strip().lower() for k in paper_row['keywords'].split(',')])
            
        # Extract from title and abstract
        key_terms = [
            '3d', 'nerf', 'neural radiance field', 'diffusion', 'gaussian', 
            'splatting', 'reconstruction', 'rendering', 'text-to-3d', 
            'image-to-3d', 'mesh', 'point cloud', 'implicit surface',
            '3dgs', 'slam', 'multi-view', 'physics', 'simulation'
        ]
        
        title = paper_row.get('title', '').lower()
        abstract = paper_row.get('abstract', '').lower()
        
        for term in key_terms:
            if term in title or term in abstract:
                keywords.add(term)
                
        return list(keywords)
        
    def search_papers(self, query=None, days=None, category=None, has_code=None, limit=1000, offset=0):
        """搜索论文"""
        try:
            logger.info(f"Searching papers with params: query='{query}', days={days}, "
                       f"category='{category}', has_code={has_code}, limit={limit}, offset={offset}")
            
            conn = sqlite3.connect(self.db_path)
            logger.debug(f"Connected to database: {self.db_path}")
            
            # 构建基础查询，始终关联分类表和关键点表
            query_parts = ["SELECT DISTINCT p.*, pc.category, pc.confidence, GROUP_CONCAT(pkp.point) as key_points"]
            from_parts = ["FROM papers p"]
            from_parts.append("LEFT JOIN paper_categories pc ON p.id = pc.paper_id")
            from_parts.append("LEFT JOIN paper_key_points pkp ON p.id = pkp.paper_id")
            where_parts = ["1=1"]
            params = []
            
            # 添加分类过滤
            if category:
                where_parts.append("pc.category = ?")
                params.append(category)
            
            # 添加关键词搜索
            if query:
                where_parts.append("""
                    (p.title LIKE ? OR p.abstract LIKE ? OR 
                    EXISTS (SELECT 1 FROM paper_keywords pk 
                           WHERE pk.paper_id = p.id AND pk.keyword LIKE ?))
                """)
                query_param = f"%{query}%"
                params.extend([query_param] * 3)
            
            # 添加时间过滤
            if days:
                where_parts.append("p.published >= date('now', '-' || ? || ' days')")
                params.append(days)
            
            # 添加代码过滤
            if has_code is not None:
                where_parts.append("p.has_code = ?")
                params.append(1 if has_code else 0)
            
            # 组合SQL，添加 GROUP BY
            sql = f"""
                {' '.join(query_parts)}
                {' '.join(from_parts)}
                WHERE {' AND '.join(where_parts)}
                GROUP BY p.id
                ORDER BY p.published DESC, p.citations DESC
                LIMIT ? OFFSET ?
            """
            params.append(limit)
            params.append(offset)
            
            # 执行查询
            logger.debug(f"Executing SQL: {sql}")
            logger.debug(f"Parameters: {params}")
            df = pd.read_sql_query(sql, conn, params=params)
            logger.info(f"Query returned {len(df)} rows")
            
            # 确保返回的数据包含所有必要字段
            df = df.fillna({
                'title': '',
                'authors': '',
                'abstract': '',
                'category': 'Uncategorized',
                'confidence': 0.0,
                'has_code': False,
                'citations': 0,
                'summary': '',
                'key_points': ''
            })
            
            # 修改 key_points 处理逻辑
            def process_key_points(x):
                if pd.isna(x) or not x:
                    return []
                if isinstance(x, list):
                    return x
                try:
                    if isinstance(x, str):
                        if not x.strip():
                            return []
                        return x.split(',')
                    return []
                except:
                    return []
            
            df['key_points'] = df['key_points'].apply(process_key_points)
            
            conn.close()
            return df
            
        except Exception as e:
            logger.error(f"Error in search_papers: {e}", exc_info=True)
            return pd.DataFrame()
        
    def get_paper_key_points(self, paper_id):
        """Get key points for a paper"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT point FROM paper_key_points WHERE paper_id = ?',
            (paper_id,)
        )
        
        points = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        return points
        
    def get_papers_for_subscription(self, topic, days=1):
        """
        Get papers for a specific subscription topic
        
        Args:
            topic (str): Subscription topic
            days (int): Number of days to look back
            
        Returns:
            pandas.DataFrame: DataFrame containing paper information
        """
        # Map topic to search query or category
        topic_mapping = {
            'nerf': ('nerf OR "neural radiance field"', 'Neural Rendering'),
            'diffusion': ('diffusion', 'Generative 3D'),
            'gaussian': ('gaussian', 'Neural Rendering'),
            'reconstruction': ('reconstruction', '3D Reconstruction'),
            'text-to-3d': ('text-to-3d OR "text to 3d"', 'Generative 3D'),
            'physical-simulation': ('physics OR simulation', 'Physical Simulation'),
            'all': (None, None)
        }
        
        query, category = topic_mapping.get(topic.lower(), (topic, None))
        return self.search_papers(query=query, category=category, days=days, limit=100)
    
    def get_active_subscriptions(self):
        """
        Get all active subscriptions
        
        Returns:
            list: List of subscription dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # This enables name-based access to columns
        
        cursor = conn.cursor()
        cursor.execute(
            'SELECT * FROM user_subscriptions WHERE active = 1'
        )
        
        subscriptions = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return subscriptions
        
    def update_paper_metadata(self, paper_id, metadata):
        """
        Update paper metadata
        
        Args:
            paper_id (str): Paper ID
            metadata (dict): Metadata to update
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Update fields
        updates = []
        params = []
        
        for key, value in metadata.items():
            if key in ['code_url', 'project_url', 'video_url', 'citations', 'social_score', 'summary']:
                updates.append(f"{key} = ?")
                params.append(value)
                
        if 'code_url' in metadata and metadata['code_url']:
            updates.append("has_code = 1")
            
        if not updates:
            return
            
        # Execute update
        sql = f"UPDATE papers SET {', '.join(updates)} WHERE id = ?"
        params.append(paper_id)
        
        cursor.execute(sql, params)
        conn.commit()
        conn.close()
        
    def add_classification(self, paper_id, category, confidence=1.0):
        """
        Add classification to a paper
        
        Args:
            paper_id (str): Paper ID
            category (str): Category (e.g., 'Neural Rendering', 'Generative 3D')
            confidence (float): Confidence score (0-1)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Insert or replace classification
        cursor.execute(
            'INSERT OR REPLACE INTO paper_categories (paper_id, category, confidence) VALUES (?, ?, ?)',
            (paper_id, category, confidence)
        )
        
        conn.commit()
        conn.close()
    
    def add_key_points(self, paper_id, key_points):
        """
        Add key points to a paper
        
        Args:
            paper_id (str): Paper ID
            key_points (list): List of key points
        """
        if not key_points:
            return
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Delete existing key points
        cursor.execute(
            'DELETE FROM paper_key_points WHERE paper_id = ?',
            (paper_id,)
        )
        
        # Insert new key points
        for point in key_points:
            cursor.execute(
                'INSERT INTO paper_key_points (paper_id, point) VALUES (?, ?)',
                (paper_id, point)
            )
            
        conn.commit()
        conn.close()
    
    def add_subscription(self, user_id, topic, frequency='daily', format='email', email=None, webhook_url=None, wechat_id=None):
        """
        Add a new subscription
        
        Args:
            user_id (str): User ID
            topic (str): Topic to subscribe to
            frequency (str): Frequency of notifications ('daily', 'weekly', 'monthly')
            format (str): Notification format ('email', 'webhook', 'wechat')
            email (str): Email address for email notifications
            webhook_url (str): Webhook URL for webhook notifications
            wechat_id (str): WeChat ID for WeChat notifications
            
        Returns:
            int: Subscription ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            '''
            INSERT INTO user_subscriptions 
            (user_id, topic, frequency, format, email, webhook_url, wechat_id, created_at, active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
            ''',
            (user_id, topic, frequency, format, email, webhook_url, wechat_id, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        )
        
        subscription_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return subscription_id
    
    def update_subscription_last_sent(self, subscription_id):
        """
        Update the last sent time for a subscription
        
        Args:
            subscription_id (int): Subscription ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            'UPDATE user_subscriptions SET last_sent = ? WHERE id = ?',
            (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), subscription_id)
        )
        
        conn.commit()
        conn.close()
    
    def deactivate_subscription(self, subscription_id):
        """
        Deactivate a subscription
        
        Args:
            subscription_id (int): Subscription ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            'UPDATE user_subscriptions SET active = 0 WHERE id = ?',
            (subscription_id,)
        )
        
        conn.commit()
        conn.close()
    
    def get_top_keywords(self, days=30, limit=20):
        """
        Get the top keywords for papers in the last N days
        
        Args:
            days (int): Number of days to look back
            limit (int): Maximum number of keywords to return
            
        Returns:
            list: List of (keyword, count) tuples
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        date_cutoff = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime('%Y-%m-%d')
        
        cursor.execute(
            '''
            SELECT pk.keyword, COUNT(*) as count
            FROM paper_keywords pk
            JOIN papers p ON pk.paper_id = p.id
            WHERE p.published >= ?
            GROUP BY pk.keyword
            ORDER BY count DESC
            LIMIT ?
            ''',
            (date_cutoff, limit)
        )
        
        keywords = [(row[0], row[1]) for row in cursor.fetchall()]
        conn.close()
        
        return keywords
    
    def get_unclassified_papers(self, days=7, limit=100):
        """
        Get papers without classification
        
        Args:
            days (int): Number of days to look back
            limit (int): Maximum number of papers to return
            
        Returns:
            pandas.DataFrame: DataFrame containing paper information
        """
        conn = sqlite3.connect(self.db_path)
        
        sql = '''
        SELECT p.* FROM papers p
        WHERE p.id NOT IN (
            SELECT paper_id FROM paper_categories
        )
        AND p.published >= ?
        ORDER BY p.published DESC
        LIMIT ?
        '''
        
        date_cutoff = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime('%Y-%m-%d')
        params = [date_cutoff, limit]
        
        df = pd.read_sql_query(sql, conn, params=params)
        conn.close()
        
        return df
    
    def get_unenriched_papers(self, days=14, limit=100):
        """
        Get papers without code links
        
        Args:
            days (int): Number of days to look back
            limit (int): Maximum number of papers to return
            
        Returns:
            pandas.DataFrame: DataFrame containing paper information
        """
        conn = sqlite3.connect(self.db_path)
        
        sql = '''
        SELECT p.* FROM papers p
        WHERE (p.code_url IS NULL OR p.code_url = '')
        AND p.has_code = 0
        AND p.published >= ?
        ORDER BY p.published DESC
        LIMIT ?
        '''
        
        date_cutoff = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime('%Y-%m-%d')
        params = [date_cutoff, limit]
        
        df = pd.read_sql_query(sql, conn, params=params)
        conn.close()
        
        return df
    
    def save_trend_report(self, report):
        """
        Save a trend report
        
        Args:
            report (dict): Trend report data
            
        Returns:
            int: Report ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            '''
            INSERT INTO trend_reports 
            (report_date, period_days, total_papers, category_distribution, top_keywords, has_code_ratio, report_data)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''',
            (
                datetime.datetime.now().strftime('%Y-%m-%d'),
                report.get('period_days', 30),
                report.get('total_papers', 0),
                json.dumps(report.get('category_distribution', {})),
                json.dumps(report.get('top_keywords', [])),
                report.get('has_code_ratio', 0),
                json.dumps(report)
            )
        )
        
        report_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return report_id
    
    def get_latest_trend_report(self):
        """
        Get the latest trend report
        
        Returns:
            dict: Trend report data
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            '''
            SELECT report_data FROM trend_reports
            ORDER BY report_date DESC
            LIMIT 1
            '''
        )
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return json.loads(result[0])
        else:
            return None
    
    def get_paper_details(self, paper_id):
        """
        Get detailed information for a specific paper
        
        Args:
            paper_id (str): Paper ID
            
        Returns:
            dict: Paper details
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get paper data
        cursor.execute(
            'SELECT * FROM papers WHERE id = ?',
            (paper_id,)
        )
        paper = cursor.fetchone()
        
        if not paper:
            conn.close()
            return None
            
        paper_dict = dict(paper)
        
        # Get categories
        cursor.execute(
            'SELECT category, confidence FROM paper_categories WHERE paper_id = ? ORDER BY confidence DESC',
            (paper_id,)
        )
        paper_dict['categories'] = [dict(row) for row in cursor.fetchall()]
        
        # Get keywords
        cursor.execute(
            'SELECT keyword FROM paper_keywords WHERE paper_id = ?',
            (paper_id,)
        )
        paper_dict['keywords'] = [row[0] for row in cursor.fetchall()]
        
        # Get key points
        cursor.execute(
            'SELECT point FROM paper_key_points WHERE paper_id = ?',
            (paper_id,)
        )
        paper_dict['key_points'] = [row[0] for row in cursor.fetchall()]
        
        conn.close()
        
        return paper_dict
    
    def get_statistics(self):
        """
        Get database statistics
        
        Returns:
            dict: Statistics dictionary
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stats = {}
        
        # Total papers
        cursor.execute('SELECT COUNT(*) FROM papers')
        stats['total_papers'] = cursor.fetchone()[0]
        
        # Papers by source
        cursor.execute('SELECT source, COUNT(*) FROM papers GROUP BY source')
        stats['papers_by_source'] = dict(cursor.fetchall())
        
        # Papers by category
        cursor.execute('''
            SELECT category, COUNT(*) 
            FROM paper_categories 
            GROUP BY category
            ORDER BY COUNT(*) DESC
        ''')
        stats['papers_by_category'] = dict(cursor.fetchall())
        
        # Papers with code
        cursor.execute('SELECT COUNT(*) FROM papers WHERE has_code = 1')
        stats['papers_with_code'] = cursor.fetchone()[0]
        
        # Papers by date (last 60 days)
        cursor.execute('''
            SELECT published, COUNT(*) 
            FROM papers 
            WHERE published >= date('now', '-60 days')
            GROUP BY published
            ORDER BY published
        ''')
        stats['papers_by_date'] = dict(cursor.fetchall())
        
        # Active subscriptions
        cursor.execute('SELECT COUNT(*) FROM user_subscriptions WHERE active = 1')
        stats['active_subscriptions'] = cursor.fetchone()[0]
        
        conn.close()
        
        return stats

# Example usage
if __name__ == "__main__":
    # Initialize database manager
    db_manager = PaperDatabaseManager()
    
    # Print database statistics
    stats = db_manager.get_statistics()
    print(f"Total papers: {stats['total_papers']}")
    print(f"Papers with code: {stats['papers_with_code']}")
    print(f"Active subscriptions: {stats['active_subscriptions']}")
    
    # Example: Add a test subscription
    # subscription_id = db_manager.add_subscription(
    #     user_id="test_user",
    #     topic="nerf",
    #     frequency="daily",
    #     format="email",
    #     email="test@example.com"
    # )
    # print(f"Added subscription with ID: {subscription_id}")
