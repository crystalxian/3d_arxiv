import os
import smtplib
import json
import datetime
import requests
import pandas as pd
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from jinja2 import Template

class NotificationSystem:
    def __init__(self, email_config=None):
        """
        Initialize the notification system
        
        Args:
            email_config (dict): Email configuration
        """
        self.email_config = email_config or {
            'smtp_server': os.environ.get('SMTP_SERVER', 'smtp.gmail.com'),
            'smtp_port': int(os.environ.get('SMTP_PORT', 587)),
            'username': os.environ.get('EMAIL_USERNAME'),
            'password': os.environ.get('EMAIL_PASSWORD'),
            'sender': os.environ.get('EMAIL_SENDER')
        }
    
    def send_email(self, recipient, subject, html_content):
        """
        Send email notification
        
        Args:
            recipient (str): Recipient email address
            subject (str): Email subject
            html_content (str): HTML content of the email
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.email_config['username'] or not self.email_config['password']:
            print("Email configuration incomplete. Skipping email sending.")
            return False
        
        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.email_config['sender'] or self.email_config['username']
            msg['To'] = recipient
            
            # Attach HTML content
            html_part = MIMEText(html_content, 'html')
            msg.attach(html_part)
            
            # Connect to SMTP server and send
            with smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port']) as server:
                server.starttls()
                server.login(self.email_config['username'], self.email_config['password'])
                server.send_message(msg)
                
            print(f"Email sent to {recipient}")
            return True
            
        except Exception as e:
            print(f"Error sending email: {e}")
            return False
    
    def send_webhook(self, webhook_url, data):
        """
        Send webhook notification
        
        Args:
            webhook_url (str): Webhook URL
            data (dict): Data to send
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            response = requests.post(
                webhook_url,
                json=data,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code >= 200 and response.status_code < 300:
                print(f"Webhook sent successfully to {webhook_url}")
                return True
            else:
                print(f"Webhook error: {response.status_code} {response.text}")
                return False
                
        except Exception as e:
            print(f"Error sending webhook: {e}")
            return False
    
    def generate_email_content(self, papers_df, topic=None):
        """
        Generate HTML email content from papers dataframe
        
        Args:
            papers_df (pandas.DataFrame): DataFrame containing papers
            topic (str): Topic of the papers
            
        Returns:
            str: HTML content
        """
        # Define email template
        template_str = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body { font-family: Arial, sans-serif; line-height: 1.5; max-width: 800px; margin: 0 auto; }
                .paper { margin-bottom: 20px; border-bottom: 1px solid #eee; padding-bottom: 15px; }
                .title { font-size: 18px; font-weight: bold; color: #333; }
                .authors { color: #666; font-style: italic; margin-bottom: 5px; }
                .abstract { color: #444; margin-bottom: 10px; }
                .links { margin-top: 10px; }
                .links a { color: #0066cc; text-decoration: none; margin-right: 15px; }
                .category { display: inline-block; padding: 3px 8px; background: #f0f0f0; border-radius: 3px; font-size: 12px; }
                .header { margin-bottom: 20px; border-bottom: 2px solid #333; padding-bottom: 10px; }
                .footer { margin-top: 30px; font-size: 12px; color: #666; border-top: 1px solid #eee; padding-top: 10px; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>3DGen Daily Digest - {{ topic }}</h1>
                <p>{{ date }} | {{ count }} new papers</p>
            </div>
            
            {% for paper in papers %}
            <div class="paper">
                <div class="title">{{ paper.title }}</div>
                <div class="authors">{{ paper.authors }}</div>
                <div class="category">{{ paper.category }}</div>
                <div class="abstract">{{ paper.summary }}</div>
                <div class="links">
                    <a href="{{ paper.pdf_url }}" target="_blank">PDF</a>
                    {% if paper.code_url %}
                    <a href="{{ paper.code_url }}" target="_blank">Code</a>
                    {% endif %}
                    {% if paper.project_url %}
                    <a href="{{ paper.project_url }}" target="_blank">Project</a>
                    {% endif %}
                    {% if paper.video_url %}
                    <a href="{{ paper.video_url }}" target="_blank">Demo</a>
                    {% endif %}
                </div>
            </div>
            {% endfor %}
            
            <div class="footer">
                <p>You received this email because you're subscribed to {{ topic }} updates from 3DGenPapersBot.</p>
                <p>To unsubscribe or modify your preferences, reply with "unsubscribe" or visit your account settings.</p>
            </div>
        </body>
        </html>
        """
        
        template = Template(template_str)
        
        # Format the data
        papers_list = []
        for _, paper in papers_df.iterrows():
            paper_dict = paper.to_dict()
            
            # Convert NaN values to None
            for key, value in paper_dict.items():
                if pd.isna(value):
                    paper_dict[key] = None
            
            papers_list.append(paper_dict)
            
        # Render the template
        html_content = template.render(
            papers=papers_list,
            topic=topic or "AI 3D Generation",
            date=datetime.datetime.now().strftime("%Y-%m-%d"),
            count=len(papers_df)
        )
        
        return html_content
    
    def generate_webhook_payload(self, papers_df, topic=None):
        """
        Generate webhook payload from papers dataframe
        
        Args:
            papers_df (pandas.DataFrame): DataFrame containing papers
            topic (str): Topic of the papers
            
        Returns:
            dict: Webhook payload
        """
        papers_list = []
        for _, paper in papers_df.iterrows():
            paper_dict = {
                'id': paper.get('id'),
                'title': paper.get('title'),
                'authors': paper.get('authors'),
                'summary': paper.get('summary') if 'summary' in paper else paper.get('abstract'),
                'category': paper.get('category'),
                'pdf_url': paper.get('pdf_url'),
                'code_url': paper.get('code_url'),
                'published': paper.get('published')
            }
            
            # Convert dates to strings for JSON serialization
            if isinstance(paper_dict['published'], (datetime.date, datetime.datetime)):
                paper_dict['published'] = paper_dict['published'].isoformat()
                
            papers_list.append(paper_dict)
            
        return {
            'topic': topic or "AI 3D Generation",
            'date': datetime.datetime.now().isoformat(),
            'count': len(papers_df),
            'papers': papers_list
        }
    
    def send_subscription_notification(self, subscription, papers_df, db_manager=None):
        """
        Send notification for a subscription
        
        Args:
            subscription (dict): Subscription information
            papers_df (pandas.DataFrame): DataFrame containing papers
            db_manager: Optional database manager for updating last sent time
            
        Returns:
            bool: True if successful, False otherwise
        """
        if papers_df.empty:
            print(f"No papers to send for subscription {subscription['id']}")
            return False
            
        topic = subscription.get('topic', 'AI 3D Generation')
        format = subscription.get('format', 'email')
        user_id = subscription.get('user_id')
        
        success = False
        
        if format == 'email':
            # Send email notification
            email = subscription.get('email')
            if not email:
                print(f"No email address for subscription {subscription['id']}")
                return False
                
            html_content = self.generate_email_content(papers_df, topic)
            subject = f"3DGen Daily Digest - {topic} ({datetime.datetime.now().strftime('%Y-%m-%d')})"
            
            success = self.send_email(email, subject, html_content)
            
        elif format == 'webhook':
            # Send webhook notification
            webhook_url = subscription.get('webhook_url')
            if not webhook_url:
                print(f"No webhook URL for subscription {subscription['id']}")
                return False
                
            payload = self.generate_webhook_payload(papers_df, topic)
            success = self.send_webhook(webhook_url, payload)
            
        elif format == 'wechat':
            # Send WeChat notification
            # This would require integration with WeChat API
            print("WeChat notification not implemented yet")
            return False
            
        # Update last sent time in database if successful
        if success and db_manager:
            db_manager.update_subscription_last_sent(subscription['id'])
            
        return success

# Example usage
if __name__ == "__main__":
    from database_manager import PaperDatabaseManager
    
    # Initialize notification system
    notifier = NotificationSystem()
    
    # Initialize database manager
    db_manager = PaperDatabaseManager()
    
    # Get recent papers for a topic
    papers_df = db_manager.get_papers_for_subscription('nerf', days=1)
    
    # Example subscription
    subscription = {
        'id': 1,
        'user_id': 'user123',
        'topic': 'NeRF & Neural Rendering',
        'format': 'email',
        'email': 'example@example.com'  # Replace with actual email for testing
    }
    
    # Send notification
    if not papers_df.empty:
        notifier.send_subscription_notification(subscription, papers_df, db_manager)
    else:
        print("No new papers found")
