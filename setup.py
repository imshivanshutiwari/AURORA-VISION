from setuptools import setup, find_packages

setup(
    name="aurora-vision",
    version="1.0.0",
    description="Multi-Modal Video Understanding System: Visual Frames + Audio Transcript + On-Screen Text",
    author="AURORA-VISION Team",
    python_requires=">=3.11",
    packages=find_packages(),
    install_requires=[
        "torch>=2.3.0",
        "torchvision>=0.18.0",
        "transformers>=4.40.0",
        "plotly>=5.22.0",
        "dash>=2.17.0",
        "langchain>=0.1.20",
        "langgraph>=0.0.55",
        "numpy>=1.26.4",
        "pandas>=2.2.2",
        "pydantic>=2.7.1",
        "pyyaml>=6.0.1",
        "fastapi>=0.111.0",
        "uvicorn>=0.29.0",
    ],
    entry_points={
        "console_scripts": [
            "aurora-vision=pipeline.main:main",
            "aurora-dashboard=dashboard.app:main",
        ]
    },
)
