# import os
# from pathlib import Path
# import logging

# logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(asctime)s:')


# list_of_files = [
#     "src/__init__.py",
#     "src/helper.py",
#     "src/prompt.py",
#     ".env",
#     "setup.py",
#     "app.py",
#     "research/trials.ipynb"
# ]
# # //look through the list one by one

# for filepath in list_of_files:
#     filepath = Path(filepath)
#     filedir, filename = os.path.split(filepath)

#     if filedir != "":
#         os.makedirs(filedir, exist_ok=True)
#         logging.info(f"Creating directory: {filedir} for the file: {filename}")

#     if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
#         with open(filepath, "w") as f:
#             pass
#             logging.info(f"Creating empty file: {filepath}")
#     else:
#         logging.info(f"{filename} is already exists")

import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

list_of_files = [
    "src/__init__.py",
    "src/helper.py",
    "src/prompt.py",
    ".env",
    "setup.py",
    "app.py",
    "research/trials.ipynb"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file: {filename}")

    if (not filepath.exists()) or (filepath.stat().st_size == 0):
        with open(filepath, "w") as f:
            pass  # File created, do nothing else for now
        logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists")
