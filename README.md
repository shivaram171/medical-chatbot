# medical-chatbot


# End-to-end-Medical-Chatbot-Generative-AI

## How to run?
### STEPS:

Clone the repository

```bash
Project repo: https://github.com/
```

### STEP 01- Create a conda environment after opening the repository
```bash
conda create -n medibot python=3.10 -y
```
```bash
conda activate llmapp
```
### STEP 02- install the requirements
```bash

pip install -r requirements.txt
```
### create a '.env' file in the root directory and add your pinecone & openai credentials as follows:


```ini
PINCODE_API_KEY ="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
TOGETHER_API_KEY="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```
```bash
python store_index.py
```

```bash
python app.py
```
Now,
```bash
open up localhost:
```
### Techstack Used:

-python
-LangChain
-Flask
-Pinecone
