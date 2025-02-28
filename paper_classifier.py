import os
import json
import requests
import pandas as pd
from tqdm import tqdm
import time

class PaperClassifier:
    def __init__(self, api_key=None, model="deepseek-chat"):
        """
        Initialize the paper classifier and summarizer
        
        Args:
            api_key (str): API key for the LLM service
            model (str): LLM model to use
        """
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        self.model = model
        self.categories = [
            "Generative 3D", 
            "Neural Rendering", 
            "3D Reconstruction", 
            "Physical Simulation"
        ]
        
        if not self.api_key:
            print("Warning: No API key provided for LLM. Classification and summarization will be limited.")
    
    def classify_paper(self, title, abstract):
        """
        Classify a paper into one of the predefined categories
        
        Args:
            title (str): Paper title
            abstract (str): Paper abstract
            
        Returns:
            tuple: (category, confidence)
        """
        if not self.api_key:
            # Fallback to keyword-based classification
            return self._keyword_classify(title, abstract)
        
        prompt = f"""
        You are a research paper classifier specializing in 3D generation and computer vision.
        
        Classify the following research paper into ONE of these categories:
        1. Generative 3D (text-to-3D, image-to-3D, diffusion models for 3D, etc.)
        2. Neural Rendering (NeRF, Gaussian Splatting, novel view synthesis, etc.)
        3. 3D Reconstruction (SLAM, multi-view reconstruction, depth estimation, etc.)
        4. Physical Simulation (material generation, physics dynamics, etc.)
        
        Paper Title: {title}
        Paper Abstract: {abstract}
        
        Return ONLY the category name and confidence score (0-1) in JSON format like this:
        {{
            "category": "category_name",
            "confidence": 0.95
        }}
        """
        
        try:
            response = self._call_llm_api(prompt)
            result = json.loads(response)
            return result["category"], result["confidence"]
        except Exception as e:
            print(f"Error classifying paper: {e}")
            return self._keyword_classify(title, abstract)
    
    def generate_summary(self, title, abstract):
        """
        Generate a concise summary of the paper highlighting key contributions
        
        Args:
            title (str): Paper title
            abstract (str): Paper abstract
            
        Returns:
            str: Summary text
        """
        if not self.api_key:
            return abstract[:150] + "..." if len(abstract) > 150 else abstract
        
        prompt = f"""
        You are a research paper summarizer specializing in 3D generation research.
        
        Create a concise summary (2-3 sentences) of the following research paper, 
        highlighting the key technical contribution and practical impact.
        
        Paper Title: {title}
        Paper Abstract: {abstract}
        
        Your summary should be clear, technical, and identify what makes this paper novel.
        """
        
        try:
            return self._call_llm_api(prompt)
        except Exception as e:
            print(f"Error generating summary: {e}")
            return abstract[:150] + "..." if len(abstract) > 150 else abstract
    
    def extract_key_points(self, title, abstract):
        """
        Extract key points from the paper
        
        Args:
            title (str): Paper title
            abstract (str): Paper abstract
            
        Returns:
            list: List of key points
        """
        if not self.api_key:
            return ["No API key provided for extracting key points"]
        
        prompt = f"""
        You are a research paper analyzer specializing in 3D generation and computer vision.
        
        Extract 2-3 key technical highlights from this research paper:
        
        Paper Title: {title}
        Paper Abstract: {abstract}
        
        Return ONLY a JSON array of strings, each describing a specific technical contribution:
        [
            "First key technical highlight",
            "Second key technical highlight",
            "Third key technical highlight (if applicable)"
        ]
        """
        
        try:
            response = self._call_llm_api(prompt)
            return json.loads(response)
        except Exception as e:
            print(f"Error extracting key points: {e}")
            return ["Could not extract key points"]
    
    def process_papers_batch(self, papers_df, db_manager):
        """
        Process a batch of papers for classification and summary
        
        Args:
            papers_df (pandas.DataFrame): DataFrame containing papers
            db_manager: Database manager instance for storing results
            
        Returns:
            pandas.DataFrame: Processed papers with classifications and summaries
        """
        results = []
        
        for _, paper in tqdm(papers_df.iterrows(), total=len(papers_df), desc="Processing papers"):
            paper_id = paper['id']
            title = paper['title']
            abstract = paper['abstract']
            
            # Classify paper
            category, confidence = self.classify_paper(title, abstract)
            
            # Generate summary
            summary = self.generate_summary(title, abstract)
            
            # Extract key points
            key_points = self.extract_key_points(title, abstract)
            
            # Update database if provided
            if db_manager:
                db_manager.add_classification(paper_id, category, confidence)
            
            # Add results to output
            result = paper.to_dict()
            result['category'] = category
            result['confidence'] = confidence
            result['summary'] = summary
            result['key_points'] = key_points
            
            results.append(result)
            
            # Rate limiting for API calls
            time.sleep(1)
        
        return pd.DataFrame(results)
    
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
            "max_tokens": 500
        }
        
        # Replace this URL with the actual API endpoint
        url = "https://api.deepseek.com/v1/chat/completions"
        
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code != 200:
            raise Exception(f"API error: {response.status_code} {response.text}")
            
        return response.json()["choices"][0]["message"]["content"].strip()
    
    def _keyword_classify(self, title, abstract):
        """Fallback keyword-based classification"""
        text = (title + " " + abstract).lower()
        
        # Simple keyword matching
        keywords = {
            "Generative 3D": ["text-to-3d", "image-to-3d", "diffusion", "generation", "gan", "generative"],
            "Neural Rendering": ["nerf", "neural radiance field", "gaussian splatting", "3dgs", "novel view", "rendering"],
            "3D Reconstruction": ["reconstruction", "slam", "structure from motion", "depth", "multi-view"],
            "Physical Simulation": ["physics", "simulation", "material", "fluid", "dynamics"]
        }
        
        scores = {category: 0 for category in keywords}
        
        for category, terms in keywords.items():
            for term in terms:
                if term in text:
                    scores[category] += 1
        
        # Find the category with the highest score
        max_category = max(scores, key=scores.get)
        
        # Calculate confidence (normalized score)
        total_score = sum(scores.values())
        confidence = scores[max_category] / total_score if total_score > 0 else 0.5
        
        return max_category, min(confidence, 0.9)  # Cap confidence at 0.9 for keyword-based

# Example usage
if __name__ == "__main__":
    from database_manager import PaperDatabaseManager
    
    # Initialize classifier and database manager
    classifier = PaperClassifier()
    db_manager = PaperDatabaseManager()
    
    # Get recent papers
    papers_df = db_manager.search_papers(days=7)
    
    # Process papers
    processed_df = classifier.process_papers_batch(papers_df, db_manager)
    
    # Print results
    print(f"Processed {len(processed_df)} papers")
    print(processed_df[['title', 'category', 'confidence', 'summary']].head())
