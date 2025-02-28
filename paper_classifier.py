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
            # 清理和格式化返回的内容
            response = response.replace("'", '"').strip('`')
            if response.startswith('json'):
                response = response[4:].strip()
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
                
                # 更强健的 JSON 解析
                try:
                    # 首先尝试直接解析
                    return json.loads(response)
                except json.JSONDecodeError:
                    # 如果失败，尝试清理和格式化
                    cleaned_response = response.replace("'", '"').strip('`')
                    
                    # 移除可能的代码块标记
                    if cleaned_response.startswith('```json'):
                        cleaned_response = cleaned_response[7:]
                    elif cleaned_response.startswith('```'):
                        cleaned_response = cleaned_response[3:]
                        
                    # 移除结尾的代码块标记
                    if cleaned_response.endswith('```'):
                        cleaned_response = cleaned_response[:-3]
                        
                    cleaned_response = cleaned_response.strip()
                    
                    # 移除可能的 json 标记
                    if cleaned_response.startswith('json'):
                        cleaned_response = cleaned_response[4:].strip()
                        
                    # 如果不是以 [ 开头，尝试提取 [ ] 之间的内容
                    if not cleaned_response.startswith('['):
                        start = cleaned_response.find('[')
                        end = cleaned_response.rfind(']')
                        if start != -1 and end != -1:
                            cleaned_response = cleaned_response[start:end+1]
                    
                    try:
                        return json.loads(cleaned_response)
                    except json.JSONDecodeError:
                        # 如果仍然失败，尝试手动解析
                        # 提取引号内的文本作为关键点
                        import re
                        points = re.findall(r'"([^"]*)"', cleaned_response)
                        if points:
                            return points
                        
                        # 最后的回退方案
                        lines = [line.strip() for line in cleaned_response.split('\n') if line.strip()]
                        if lines:
                            # 移除可能的序号和特殊字符
                            points = [re.sub(r'^[\d\.\-\*]+\s*', '', line) for line in lines]
                            return points[:3]  # 最多返回3个要点
                        
                        return ["Could not extract key points"]
            except Exception as e:
                print(f"Error extracting key points: {e}")
                return ["Could not extract key points"]
        except Exception as e:
            print(f"Error extracting key points: {e}")
            return ["Could not extract key points"]
    
    def process_papers_batch(self, papers_df, db_manager):
        """Process a batch of papers for classification and summary"""
        results = []
        
        for _, paper in tqdm(papers_df.iterrows(), total=len(papers_df), desc="Processing papers"):
            try:
                paper_id = paper['id']
                title = paper['title']
                abstract = paper['abstract']
                
                # 分类处理
                try:
                    category, confidence = self.classify_paper(title, abstract)
                except Exception as e:
                    logger.error(f"Error classifying paper {paper_id}: {e}")
                    category, confidence = self._keyword_classify(title, abstract)
                
                # 生成摘要
                try:
                    summary = self.generate_summary(title, abstract)
                except Exception as e:
                    logger.error(f"Error generating summary for paper {paper_id}: {e}")
                    summary = abstract[:150] + "..." if len(abstract) > 150 else abstract
                
                # 提取关键点
                try:
                    key_points = self.extract_key_points(title, abstract)
                except Exception as e:
                    logger.error(f"Error extracting key points for paper {paper_id}: {e}")
                    key_points = ["Failed to extract key points"]
                
                # 更新数据库
                if db_manager:
                    try:
                        db_manager.add_classification(paper_id, category, confidence)
                        db_manager.add_key_points(paper_id, key_points)
                    except Exception as e:
                        logger.error(f"Error updating database for paper {paper_id}: {e}")
                
                # 添加结果
                result = paper.to_dict()
                result.update({
                    'category': category,
                    'confidence': confidence,
                    'summary': summary,
                    'key_points': key_points
                })
                results.append(result)
                
                # API 调用限制
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error processing paper {paper.get('id', 'unknown')}: {e}")
                continue
        
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
            "temperature": 0.1,
            "max_tokens": 500
        }
        
        url = "https://api.deepseek.com/v1/chat/completions"
        
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            
            response_data = response.json()
            if "choices" not in response_data or not response_data["choices"]:
                raise Exception("Invalid API response format")
                
            content = response_data["choices"][0]["message"]["content"].strip()
            
            # 对于需要 JSON 解析的方法，直接返回字符串内容
            return content
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {str(e)}")
        except json.JSONDecodeError as e:
            raise Exception(f"Failed to parse API response: {str(e)}")
        except Exception as e:
            raise Exception(f"API call failed: {str(e)}")
    
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
