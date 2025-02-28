from flask import Flask, request, jsonify, send_from_directory, render_template
import os
import json
import datetime
from werkzeug.middleware.proxy_fix import ProxyFix
from flask_cors import CORS
import pandas as pd
import threading
import logging
import sqlite3

# Import our modules
from database_manager import PaperDatabaseManager
from main_controller import PapersAgentController, DEFAULT_CONFIG

# 导入PaperAnalyzer和add_analyzer_endpoints
from paper_summary_comparison import PaperAnalyzer, add_analyzer_endpoints



# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api_server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize components first
db_manager = PaperDatabaseManager()
controller = PapersAgentController(DEFAULT_CONFIG)

# 在 flask_api.py 文件中

# 首先导入所有需要的模块
from flask import Flask, request, jsonify, send_from_directory, render_template
import os
import json
import datetime
from werkzeug.middleware.proxy_fix import ProxyFix
from flask_cors import CORS
import pandas as pd
import threading
import logging
import re  # 添加这行

# 导入数据库管理器
from database_manager import PaperDatabaseManager

# 导入控制器
from main_controller import PapersAgentController, DEFAULT_CONFIG

# 导入论文分析器
from paper_summary_comparison import PaperAnalyzer, add_analyzer_endpoints

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api_server.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 初始化 Flask 应用
app = Flask(__name__, 
            static_folder='static',
            template_folder='templates')
CORS(app)  # 启用 CORS
app.wsgi_app = ProxyFix(app.wsgi_app)

# 初始化数据库管理器
db_manager = PaperDatabaseManager()

# 初始化控制器
controller = PapersAgentController(DEFAULT_CONFIG)

# 初始化论文分析器
analyzer = PaperAnalyzer(db_manager=db_manager)

# 添加分析器端点
add_analyzer_endpoints(app, analyzer)

# 然后是其他 API 路由和代码...
# 初始化数据库函数
def initialize_database():
    try:
        db_manager.setup_database()
        logger.info("Database initialized successfully")
        
        # 检查数据库是否为空，如果为空则运行初始爬虫
        conn = sqlite3.connect(db_manager.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM papers")
        count = cursor.fetchone()[0]
        conn.close()
        
        if count == 0:
            logger.info("Database is empty, starting initial paper collection...")
            controller._scrape_papers()
            
    except Exception as e:
        logger.error(f"Error initializing database: {e}", exc_info=True)

# 创建应用前先初始化数据库
initialize_database()

app = Flask(__name__, 
            static_folder='static',
            template_folder='templates')
CORS(app)  # Enable CORS for all routes
app.wsgi_app = ProxyFix(app.wsgi_app)

# Helper function to convert dates to strings in JSON
def json_serialize(obj):
    if isinstance(obj, (datetime.date, datetime.datetime)):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")

@app.route('/')
def index():
    """Serve the main web interface"""
    return render_template('index.html')

@app.route('/api/papers', methods=['GET'])
def get_papers():
    """Get papers with filtering"""
    query = request.args.get('query')
    days = request.args.get('days', default=30, type=int)
    category = request.args.get('category')
    has_code = request.args.get('has_code')
    limit = request.args.get('limit', default=50, type=int)
    
    # Convert has_code to boolean if provided
    if has_code is not None:
        has_code = has_code.lower() in ['true', '1', 'yes']
    
    # Get papers from database
    try:
        papers_df = db_manager.search_papers(
            query=query,
            days=days,
            category=category,
            has_code=has_code,
            limit=limit
        )
        
        # Convert to list of dictionaries
        papers = papers_df.to_dict('records')
        
        # Clean up JSON serialization issues
        for paper in papers:
            for key, value in paper.items():
                if isinstance(value, (pd.Timestamp, datetime.date, datetime.datetime)):
                    paper[key] = value.isoformat()
                elif pd.isna(value):
                    paper[key] = None
        
        return jsonify({
            'count': len(papers),
            'papers': papers
        })
    except Exception as e:
        logger.error(f"Error getting papers: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/papers/<paper_id>', methods=['GET'])
def get_paper(paper_id):
    """Get detailed information for a specific paper"""
    try:
        paper = db_manager.get_paper_details(paper_id)
        
        if not paper:
            return jsonify({'error': 'Paper not found'}), 404
            
        # Clean up JSON serialization issues
        for key, value in paper.items():
            if isinstance(value, (pd.Timestamp, datetime.date, datetime.datetime)):
                paper[key] = value.isoformat()
            elif pd.isna(value):
                paper[key] = None
            
        return jsonify(paper)
    except Exception as e:
        logger.error(f"Error getting paper details: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/categories', methods=['GET'])
def get_categories():
    """Get all paper categories"""
    try:
        stats = db_manager.get_statistics()
        categories = stats.get('papers_by_category', {})
        
        return jsonify({
            'categories': [
                {'name': category, 'count': count}
                for category, count in categories.items()
            ]
        })
    except Exception as e:
        logger.error(f"Error getting categories: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/keywords', methods=['GET'])
def get_top_keywords():
    """Get top keywords"""
    try:
        days = request.args.get('days', default=30, type=int)
        limit = request.args.get('limit', default=20, type=int)
        
        keywords = db_manager.get_top_keywords(days=days, limit=limit)
        
        return jsonify({
            'keywords': [
                {'keyword': keyword, 'count': count}
                for keyword, count in keywords
            ]
        })
    except Exception as e:
        logger.error(f"Error getting keywords: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get database statistics"""
    try:
        stats = db_manager.get_statistics()
        
        # Clean up JSON serialization issues
        for key, value in stats.items():
            if isinstance(value, (dict)):
                for k, v in value.items():
                    if isinstance(k, (datetime.date, datetime.datetime)):
                        value[k.isoformat()] = v
                        del value[k]
        
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error getting stats: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/trends', methods=['GET'])
def get_trends():
    """Get trend report"""
    try:
        days = request.args.get('days', default=30, type=int)
        
        # Try to get latest report from database
        report = db_manager.get_latest_trend_report()
        
        # If no report or days parameter differs, generate a new one
        if not report or report.get('period_days') != days:
            report = controller.generate_trends_report(days=days)
            if report:
                db_manager.save_trend_report(report)
        
        return jsonify(report)
    except Exception as e:
        logger.error(f"Error getting trends: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/subscriptions', methods=['GET'])
def get_subscriptions():
    """Get subscriptions for a user"""
    try:
        user_id = request.args.get('user_id')
        
        if not user_id:
            return jsonify({'error': 'user_id parameter is required'}), 400
        
        # Get subscriptions for the user
        conn = db_manager.db_path and sqlite3.connect(db_manager.db_path)
        if not conn:
            return jsonify({'error': 'Database connection failed'}), 500
            
        cursor = conn.cursor()
        cursor.execute(
            'SELECT * FROM user_subscriptions WHERE user_id = ? AND active = 1',
            (user_id,)
        )
        
        # Convert to list of dictionaries
        columns = [col[0] for col in cursor.description]
        subscriptions = [dict(zip(columns, row)) for row in cursor.fetchall()]
        conn.close()
        
        return jsonify({
            'count': len(subscriptions),
            'subscriptions': subscriptions
        })
    except Exception as e:
        logger.error(f"Error getting subscriptions: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

# 添加邮件发送相关的导入
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
import os

# 添加邮件配置
# 修改邮件配置部分
SMTP_SERVER = os.environ.get('SMTP_SERVER', 'smtp.gmail.com')
SMTP_PORT = int(os.environ.get('SMTP_PORT', 587))
SMTP_USERNAME = os.environ.get('EMAIL_USERNAME')  # 修改这里
SMTP_PASSWORD = os.environ.get('EMAIL_PASSWORD')  # 修改这里

def send_confirmation_email(email, topic):
    """发送订阅确认邮件"""
    try:
        msg = MIMEMultipart()
        msg['From'] = SMTP_USERNAME
        msg['To'] = email
        msg['Subject'] = "确认您的 3DGenPapersBot 订阅"
        
        body = f"""
        感谢您订阅 3DGenPapersBot！
        
        您已成功订阅以下主题的更新：
        {topic}
        
        您将定期收到相关论文的更新通知。
        
        如需退订，请点击：[退订链接]
        
        祝您使用愉快！
        3DGenPapersBot 团队
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(msg)
            
        return True
    except Exception as e:
        logger.error(f"发送确认邮件失败: {e}")
        return False

@app.route('/api/subscriptions', methods=['POST'])
def create_subscription():
    """创建新的订阅"""
    try:
        data = request.json
        
        if not data:
            return jsonify({'error': '没有提供数据'}), 400
            
        required_fields = ['email', 'topic', 'frequency']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'缺少必要字段: {field}'}), 400
        
        # 验证邮箱格式
        if not re.match(r"[^@]+@[^@]+\.[^@]+", data['email']):
            return jsonify({'error': '无效的邮箱地址'}), 400
        
        # 创建订阅记录
        subscription_id = db_manager.add_subscription(
            email=data['email'],
            topic=data['topic'],
            frequency=data.get('frequency', 'daily')
        )
        
        # 发送确认邮件
        if send_confirmation_email(data['email'], data['topic']):
            return jsonify({
                'id': subscription_id,
                'message': '订阅创建成功，确认邮件已发送'
            })
        else:
            return jsonify({
                'id': subscription_id,
                'message': '订阅创建成功，但确认邮件发送失败'
            })
            
    except Exception as e:
        logger.error(f"创建订阅失败: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/subscriptions/<int:subscription_id>', methods=['DELETE'])
def delete_subscription(subscription_id):
    """Delete (deactivate) a subscription"""
    try:
        user_id = request.args.get('user_id')
        
        if not user_id:
            return jsonify({'error': 'user_id parameter is required'}), 400
            
        # Verify the subscription belongs to the user
        conn = sqlite3.connect(db_manager.db_path)
        cursor = conn.cursor()
        cursor.execute(
            'SELECT user_id FROM user_subscriptions WHERE id = ?',
            (subscription_id,)
        )
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return jsonify({'error': 'Subscription not found'}), 404
            
        if result[0] != user_id:
            return jsonify({'error': 'Unauthorized access'}), 403
            
        # Deactivate subscription
        db_manager.deactivate_subscription(subscription_id)
        
        return jsonify({
            'message': 'Subscription deleted successfully'
        })
    except Exception as e:
        logger.error(f"Error deleting subscription: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/trigger/scrape', methods=['POST'])
def trigger_scrape():
    """Manually trigger paper scraping"""
    try:
        # Check API key for authorization
        api_key = request.headers.get('X-API-Key')
        if not api_key or api_key != os.environ.get('API_ADMIN_KEY'):
            return jsonify({'error': 'Unauthorized'}), 403
            
        # Run scraper in background thread
        def scrape_task():
            controller._scrape_papers()
            
        thread = threading.Thread(target=scrape_task)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'message': 'Scraping started in background'
        })
    except Exception as e:
        logger.error(f"Error triggering scrape: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/trigger/classify', methods=['POST'])
def trigger_classify():
    """Manually trigger paper classification"""
    try:
        # Check API key for authorization
        api_key = request.headers.get('X-API-Key')
        if not api_key or api_key != os.environ.get('API_ADMIN_KEY'):
            return jsonify({'error': 'Unauthorized'}), 403
            
        # Run classification in background thread
        def classify_task():
            controller._classify_papers()
            
        thread = threading.Thread(target=classify_task)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'message': 'Classification started in background'
        })
    except Exception as e:
        logger.error(f"Error triggering classification: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/trigger/enrich', methods=['POST'])
def trigger_enrich():
    """Manually trigger paper enrichment"""
    try:
        # Check API key for authorization
        api_key = request.headers.get('X-API-Key')
        if not api_key or api_key != os.environ.get('API_ADMIN_KEY'):
            return jsonify({'error': 'Unauthorized'}), 403
            
        # Run enrichment in background thread
        def enrich_task():
            controller._enrich_papers()
            
        thread = threading.Thread(target=enrich_task)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'message': 'Enrichment started in background'
        })
    except Exception as e:
        logger.error(f"Error triggering enrichment: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/trigger/notify', methods=['POST'])
def trigger_notify():
    """Manually trigger notifications"""
    try:
        # Check API key for authorization
        api_key = request.headers.get('X-API-Key')
        if not api_key or api_key != os.environ.get('API_ADMIN_KEY'):
            return jsonify({'error': 'Unauthorized'}), 403
            
        # Run notification in background thread
        def notify_task():
            controller._send_notifications()
            
        thread = threading.Thread(target=notify_task)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'message': 'Notifications started in background'
        })
    except Exception as e:
        logger.error(f"Error triggering notifications: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/search', methods=['GET'])
def search():
    """Combined search endpoint for papers"""
    try:
        query = request.args.get('q', '')
        category = request.args.get('category')
        days = request.args.get('days', default=30, type=int)
        has_code = request.args.get('has_code')
        page = request.args.get('page', default=1, type=int)
        per_page = request.args.get('per_page', default=20, type=int)
        
        logger.info(f"Search request received - Parameters: query='{query}', category='{category}', "
                   f"days={days}, has_code={has_code}, page={page}, per_page={per_page}")
        
        # 获取总数和当前页数据
        total_df = db_manager.search_papers(query=query, days=days, category=category, has_code=has_code)
        total_count = len(total_df)
        logger.info(f"Total matching papers found: {total_count}")
        
        # 获取分页数据
        papers_df = db_manager.search_papers(
            query=query, 
            days=days, 
            category=category, 
            has_code=has_code,
            limit=per_page,
            offset=(page - 1) * per_page
        )
        logger.info(f"Retrieved {len(papers_df)} papers for current page")
        
        # 转换为字典列表
        papers = []
        for _, paper in papers_df.iterrows():
            paper_dict = paper.to_dict()
            # 处理日期格式
            for key in ['published', 'updated', 'date_added']:
                if key in paper_dict and pd.notnull(paper_dict[key]):
                    paper_dict[key] = pd.to_datetime(paper_dict[key]).isoformat()
            papers.append(paper_dict)
        
        response_data = {
            'total': total_count,
            'page': page,
            'per_page': per_page,
            'pages': (total_count + per_page - 1) // per_page,
            'papers': papers
        }
        logger.info(f"Sending response with {len(papers)} papers")
        logger.debug(f"Response data: {response_data}")
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error searching papers: {e}", exc_info=True)
        return jsonify({
            'error': str(e),
            'total': 0,
            'page': 1,
            'per_page': per_page,
            'pages': 0,
            'papers': []
        }), 500

@app.route('/api/wechat/verify', methods=['GET'])
def wechat_verify():
    """Verification endpoint for WeChat Official Account"""
    try:
        signature = request.args.get('signature')
        timestamp = request.args.get('timestamp')
        nonce = request.args.get('nonce')
        echostr = request.args.get('echostr')
        
        # This is a simple verification that just returns the echostr
        # In a real implementation, you should verify the signature
        return echostr
    except Exception as e:
        logger.error(f"Error in WeChat verification: {e}", exc_info=True)
        return "error", 500

@app.route('/api/wechat/message', methods=['POST'])
def wechat_message():
    """Handle incoming messages from WeChat Official Account"""
    try:
        # Parse XML message from WeChat
        xml_data = request.data
        # Process XML data and handle commands
        # This would require proper XML parsing and WeChat message handling
        
        # Return success response
        return "success"
    except Exception as e:
        logger.error(f"Error handling WeChat message: {e}", exc_info=True)
        return "success"  # Always return success to WeChat

@app.route('/api/test_db', methods=['GET'])
def test_db():
    """Test database connection and content"""
    try:
        conn = sqlite3.connect(db_manager.db_path)
        cursor = conn.cursor()
        
        # 检查表结构
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        # 获取每个表的记录数
        counts = {}
        for (table_name,) in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            counts[table_name] = cursor.fetchone()[0]
        
        conn.close()
        
        return jsonify({
            'database_path': db_manager.db_path,
            'tables': tables,
            'record_counts': counts
        })
    except Exception as e:
        logger.error(f"Database test error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

def start_scheduler():
    """Start the scheduler in a background thread"""
    controller.setup_scheduled_jobs()
    scheduler_thread = threading.Thread(target=controller.run_scheduler)
    scheduler_thread.daemon = True
    scheduler_thread.start()
    logger.info("Scheduler started in background thread")

# Auto-start scheduler when running in production
if os.environ.get('FLASK_ENV') == 'production':
    start_scheduler()

if __name__ == '__main__':
    # Parse command-line arguments
    import argparse
    parser = argparse.ArgumentParser(description='3D Generation Papers API Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--start-scheduler', action='store_true', help='Start scheduler')
    args = parser.parse_args()
    
    if args.start_scheduler:
        start_scheduler()
    
    # Run the Flask app
    app.run(host=args.host, port=args.port, debug=args.debug)
