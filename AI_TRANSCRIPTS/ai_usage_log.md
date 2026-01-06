# AI Assistance Log

This project was developed with the assistance of AI tools. Below is a summary of how AI was utilized to improve speed, quality, and documentation.

## Tools Used

- **Agentic IDE (Cursor/Windsurf equivalent)**: Used for planning, file creation, terminal command execution, and refactoring.
- **Large Language Models (Gemini/GPT-4)**: Used for:
  - Boiletplate code generation (FastAPI setup, Scikit-Learn pipelines).
  - Writing docstrings and inline documentation.
  - Debugging specific error messages.
  - Drafting the README content.

## Key Sessions

1.  **Project Initialization**: Set up directory structure, git init, and requirements.txt.
2.  **Logic Design**: Developed the deterministic hashing algorithm (`utils.py`) to ensure reproducible train/control splits without a database.
3.  **Pipeline Construction**: Built the standard scaler + OHE pipeline in `train.py`.
4.  **Refinement**: Optimized the `serve.py` endpoint to include real-time drift monitoring (PSI) and data integrity checks.
5.  **Documentation**: Generated the comprehensive README and this log.
