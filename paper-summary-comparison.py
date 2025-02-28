import os
import json
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from database_manager import PaperDatabaseManager

class PaperAnalyzer:
    def __init__(self, db_manager=None, api_key=None):
        """
        Initialize the paper analyzer
        
        Args:
            db_manager: Database manager instance
            api_key (str): API key for LLM service
        """
        self.db_manager = db_manager or PaperDatabaseManager()
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        self.model = "deepseek-ai/deepseek-coder-33b-instruct"
    
    def generate_detailed_summary(self, paper_id):
        """
        Generate a detailed summary for a paper
        
        Args:
            paper_id (str): Paper ID
            
        Returns:
            dict: Summary information
        """
        # Get paper details
        paper = self.db_manager.get_paper_details(paper_id)
        
        if not paper:
            return {"error": "Paper not found"}
        
        # If we already have a summary, return it
        if paper.get('summary') and len(paper.get('summary', '')) > 100:
            return {
                "id": paper['id'],
                "title": paper['title'],
                "authors": paper['authors'],
                "summary": paper['summary'],
                "key_points": paper.get('key_points', []),
                "source": paper['source'],
                "published": paper['published']
            }
        
        # Otherwise, generate a new summary
        if not self.api_key:
            # Fallback to simple summary
            return {
                "id": paper['id'],
                "title": paper['title'],
                "authors": paper['authors'],
                "summary": paper.get('abstract', '')[:300] + "...",
                "key_points": [],
                "source": paper['source'],
                "published": paper['published']
            }
        
        try:
            # Generate detailed summary
            prompt = f"""
            Analyze this research paper in the field of 3D generation and computer vision:
            
            Title: {paper['title']}
            Authors: {paper['authors']}
            Abstract: {paper['abstract']}
            
            Please provide:
            1. A comprehensive summary (3-4 paragraphs) explaining the paper's key contributions
            2. 3-5 key technical insights or innovations
            3. Potential applications or impact of this research
            
            Format your response as a JSON object with the following keys:
            - summary: A detailed summary of the paper
            - key_points: An array of strings, each describing a key technical insight
            - applications: An array of strings describing potential applications
            """
            
            # Call LLM API
            response = self._call_llm_api(prompt)
            result = json.loads(response)
            
            # Update database with new summary
            self.db_manager.update_paper_metadata(paper['id'], {'summary': result['summary']})
            self.db_manager.add_key_points(paper['id'], result['key_points'])
            
            return {
                "id": paper['id'],
                "title": paper['title'],
                "authors": paper['authors'],
                "summary": result['summary'],
                "key_points": result['key_points'],
                "applications": result.get('applications', []),
                "source": paper['source'],
                "published": paper['published']
            }
            
        except Exception as e:
            print(f"Error generating summary: {e}")
            return {
                "id": paper['id'],
                "title": paper['title'],
                "authors": paper['authors'],
                "summary": paper.get('abstract', '')[:300] + "...",
                "key_points": [],
                "source": paper['source'],
                "published": paper['published'],
                "error": str(e)
            }
    
    def compare_papers(self, paper_ids):
        """
        Compare multiple papers
        
        Args:
            paper_ids (list): List of paper IDs
            
        Returns:
            dict: Comparison results
        """
        if len(paper_ids) < 2:
            return {"error": "At least two papers are required for comparison"}
        
        papers = []
        for paper_id in paper_ids:
            paper = self.db_manager.get_paper_details(paper_id)
            if paper:
                papers.append(paper)
        
        if len(papers) < 2:
            return {"error": "Could not find enough papers for comparison"}
            
        if not self.api_key:
            # Fallback to simple comparison
            return {
                "papers": papers,
                "comparison": "API key required for detailed comparison"
            }
            
        try:
            # Prepare paper information
            papers_info = "\n\n".join([
                f"Paper {i+1}:\nTitle: {p['title']}\nAuthors: {p['authors']}\nAbstract: {p.get('abstract', '')}"
                for i, p in enumerate(papers)
            ])
            
            # Generate comparison
            prompt = f"""
            Compare and contrast the following research papers in the field of 3D generation:
            
            {papers_info}
            
            Please provide:
            1. A comparative analysis highlighting similarities and differences
            2. Strengths and weaknesses of each approach
            3. A table comparing key metrics or features (if applicable)
            
            Format your response as a JSON object with the following keys:
            - comparison: A detailed analysis comparing the papers
            - similarities: An array of strings describing shared aspects
            - differences: An array of strings describing key differences
            - metrics: An object with metric names as keys and arrays of values (one per paper) as values
            """
            
            # Call LLM API
            response = self._call_llm_api(prompt)
            result = json.loads(response)
            
            return {
                "papers": [
                    {
                        "id": p['id'],
                        "title": p['title'],
                        "authors": p['authors'],
                        "summary": p.get('summary', p.get('abstract', ''))[:300] + "...",
                        "published": p['published']
                    }
                    for p in papers
                ],
                "comparison": result.get('comparison', ''),
                "similarities": result.get('similarities', []),
                "differences": result.get('differences', []),
                "metrics": result.get('metrics', {})
            }
            
        except Exception as e:
            print(f"Error comparing papers: {e}")
            return {
                "papers": [
                    {
                        "id": p['id'],
                        "title": p['title'],
                        "authors": p['authors']
                    }
                    for p in papers
                ],
                "error": str(e)
            }
    
    def extract_performance_metrics(self, paper_id):
        """
        Extract performance metrics from a paper
        
        Args:
            paper_id (str): Paper ID
            
        Returns:
            dict: Extracted metrics
        """
        paper = self.db_manager.get_paper_details(paper_id)
        
        if not paper:
            return {"error": "Paper not found"}
            
        if not self.api_key or not paper.get('abstract'):
            return {"error": "Cannot extract metrics"}
            
        try:
            # Generate prompt
            prompt = f"""
            Extract quantitative performance metrics from this research paper:
            
            Title: {paper['title']}
            Abstract: {paper['abstract']}
            
            Please identify metrics such as:
            - PSNR, SSIM, LPIPS for image quality
            - FID, Inception scores for generative models
            - FPS, training time, model size for efficiency
            - Accuracy, precision, recall for evaluation
            
            Format your response as a JSON object where keys are metric names and values are the numerical results.
            Include units where applicable. If a range or multiple values are given, provide an array of values.
            """
            
            # Call LLM API
            response = self._call_llm_api(prompt)
            metrics = json.loads(response)
            
            return {
                "id": paper['id'],
                "title": paper['title'],
                "metrics": metrics
            }
            
        except Exception as e:
            print(f"Error extracting metrics: {e}")
            return {
                "id": paper['id'],
                "title": paper['title'],
                "error": str(e)
            }
    
    def generate_comparison_chart(self, paper_ids, metric_name):
        """
        Generate a comparison chart for a specific metric
        
        Args:
            paper_ids (list): List of paper IDs
            metric_name (str): Metric to compare
            
        Returns:
            str: Path to generated chart image
        """
        papers = []
        for paper_id in paper_ids:
            metrics = self.extract_performance_metrics(paper_id)
            paper = self.db_manager.get_paper_details(paper_id)
            if 'error' not in metrics and paper:
                short_title = paper['title'][:30] + "..." if len(paper['title']) > 30 else paper['title']
                if metric_name in metrics['metrics']:
                    papers.append({
                        'title': short_title,
                        'value': metrics['metrics'][metric_name]
                    })
        
        if not papers:
            return {"error": "No comparable metrics found"}
            
        # Create comparison chart
        titles = [p['title'] for p in papers]
        values = [p['value'] for p in papers]
        
        plt.figure(figsize=(10, 6))
        plt.bar(titles, values)
        plt.title(f"Comparison of {metric_name}")
        plt.ylabel(metric_name)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save chart
        chart_path = f"static/charts/comparison_{metric_name.lower().replace(' ', '_')}.png"
        os.makedirs(os.path.dirname(chart_path), exist_ok=True)
        plt.savefig(chart_path)
        plt.close()
        
        return {"chart_path": chart_path}
    
    def find_similar_papers(self, paper_id, limit=5):
        """
        Find papers similar to the given paper
        
        Args:
            paper_id (str): Paper ID
            limit (int): Maximum number of similar papers
            
        Returns:
            list: Similar papers
        """
        paper = self.db_manager.get_paper_details(paper_id)
        
        if not paper:
            return {"error": "Paper not found"}
            
        # Get paper keywords
        keywords = paper.get('keywords', [])
        category = paper.get('category')
        
        if not keywords and not category:
            return {"error": "Insufficient data for finding similar papers"}
            
        # Query database for similar papers
        query = None
        if keywords:
            query = " OR ".join(keywords)
            
        similar_papers = self.db_manager.search_papers(
            query=query,
            category=category,
            days=365,  # Look back up to a year
            limit=limit + 1  # Add one to exclude the current paper
        )
        
        # Remove the current paper
        similar_papers = similar_papers[similar_papers['id'] != paper_id]
        
        # Take top matches
        similar_papers = similar_papers.head(limit)
        
        return similar_papers.to_dict('records')
    
    def generate_paper_analysis_report(self, paper_id):
        """
        Generate a comprehensive analysis report for a paper
        
        Args:
            paper_id (str): Paper ID
            
        Returns:
            dict: Analysis report
        """
        # Get detailed summary
        summary = self.generate_detailed_summary(paper_id)
        
        if 'error' in summary:
            return summary
            
        # Get performance metrics
        metrics = self.extract_performance_metrics(paper_id)
        
        # Find similar papers
        similar_papers = self.find_similar_papers(paper_id)
        
        # Combine all information
        report = {
            "id": summary['id'],
            "title": summary['title'],
            "authors": summary['authors'],
            "summary": summary['summary'],
            "key_points": summary.get('key_points', []),
            "applications": summary.get('applications', []),
            "metrics": metrics.get('metrics', {}),
            "similar_papers": similar_papers if not isinstance(similar_papers, dict) else []
        }
        
        return report
    
    def _call_llm_api(self, prompt):
        """Call LLM API with the given prompt"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,  # Low temperature for consistent outputs
            "max_tokens": 1000
        }
        
        # Replace this URL with the actual API endpoint
        url = "https://api.deepseek.com/v1/chat/completions"
        
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code != 200:
            raise Exception(f"API error: {response.status_code} {response.text}")
            
        return response.json()["choices"][0]["message"]["content"].strip()


# API Endpoints for Flask server
def add_analyzer_endpoints(app, analyzer):
    """
    Add paper analyzer endpoints to Flask app
    
    Args:
        app: Flask app
        analyzer: PaperAnalyzer instance
    """
    @app.route('/api/papers/<paper_id>/summary', methods=['GET'])
    def get_paper_summary(paper_id):
        """Get detailed summary for a paper"""
        summary = analyzer.generate_detailed_summary(paper_id)
        return json.dumps(summary, default=str)
    
    @app.route('/api/papers/compare', methods=['POST'])
    def compare_papers():
        """Compare multiple papers"""
        data = request.json
        if not data or 'paper_ids' not in data:
            return jsonify({"error": "paper_ids required"}), 400
            
        comparison = analyzer.compare_papers(data['paper_ids'])
        return json.dumps(comparison, default=str)
    
    @app.route('/api/papers/<paper_id>/metrics', methods=['GET'])
    def get_paper_metrics(paper_id):
        """Get metrics for a paper"""
        metrics = analyzer.extract_performance_metrics(paper_id)
        return json.dumps(metrics, default=str)
    
    @app.route('/api/papers/<paper_id>/similar', methods=['GET'])
    def get_similar_papers(paper_id):
        """Get similar papers"""
        limit = request.args.get('limit', default=5, type=int)
        similar = analyzer.find_similar_papers(paper_id, limit=limit)
        return json.dumps(similar, default=str)
    
    @app.route('/api/papers/<paper_id>/analysis', methods=['GET'])
    def get_paper_analysis(paper_id):
        """Get comprehensive paper analysis"""
        analysis = analyzer.generate_paper_analysis_report(paper_id)
        return json.dumps(analysis, default=str)


# Example usage
if __name__ == "__main__":
    # Initialize database manager
    db_manager = PaperDatabaseManager()
    
    # Initialize paper analyzer
    analyzer = PaperAnalyzer(db_manager=db_manager)
    
    # Example: Generate summary for a paper
    paper_id = "arxiv_2312.04517v2"  # Replace with actual paper ID
    summary = analyzer.generate_detailed_summary(paper_id)
    print(json.dumps(summary, indent=2))
    
    # Example: Compare papers
    paper_ids = ["arxiv_2312.04517v2", "arxiv_2403.00304v1"]  # Replace with actual paper IDs
    comparison = analyzer.compare_papers(paper_ids)
    print(json.dumps(comparison, indent=2))
