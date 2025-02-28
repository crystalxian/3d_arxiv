# 3D Papers Tracking System

A system for tracking and analyzing research papers in 3D generation and computer vision.

## Features

- Daily updates from arXiv and other sources
- Automatic paper classification using LLM
- GitHub repository tracking and enrichment
- Web interface for browsing and searching papers
- Research trend analysis and visualization

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/3d_arxiv.git
cd 3d_arxiv
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

5. Initialize database:
```bash
python main_controller.py init
```

6. Run the application:
```bash
python flask_api.py
```

## Development

- Main branch: Production-ready code
- Develop branch: Development code
- Feature branches: New features and improvements

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT License - see LICENSE file for details 