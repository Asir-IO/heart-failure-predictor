<a id="readme-top"></a>

## Quick Links
- [HuggingFace Space](https://huggingface.co/spaces/Asser-IO/heart-disease-predictor)
- [Presentation Slides](https://www.canva.com/design/DAGvVv_Lz2s/zmu-fhd-StTNdbYHFa6Wmw/edit?utm_content=DAGvVv_Lz2s&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton )

<details>
<summary>Table of contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#built-with">Built With</a></li>
    <li><a href="#getting-started">Getting Started</a></li>
    <li><a href="#development-process">Development Process</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

## About The Project
This is a machine learning application that predicts heart failure death events based on patient clinical data. It uses a logistic regression model (the best fitting model).

## Built With
- Python 3.13  
- Pandas  
- NumPy  
- Scikit-Learn  
- Gradio  

## Getting Started
### Prerequisites
Ensure you have Python 3.13 and pip installed.

### Installation
1. Clone the repo:
   ```sh
   git clone https://github.com/Asir-IO/heart-failure-predictor.git
   ```
2. Create a virtual environment and install dependencies:
   ```sh
   python -m venv venv
   venv\\Scripts\\activate  # Windows
   pip install -r requirements.txt
   ```
3. Launch the Gradio app:
   ```sh
   python app.py
   ```

## Development Process
- [x] Data cleaning
- [x] Feature engineering  
- [x] Model training
- [x] Gradio app construction
- [x] Deployment on Hugging face
- [ ] Adding more intuitive ways to input data on the deployed app
- [ ] Improving the deployed app's UI
- [ ] Finding a more comprehensive dataset to increase the model accuracy

<p align="right">(<a href="#readme-top">back to top</a>)</p>
