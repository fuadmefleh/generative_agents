# Ollama Generative Agents: Interactive Simulacra of Human Behavior

Welcome to **generative_agents**, a project for building and experimenting with generative AI agents inspired by human behavior. This repository leverages the Ollama platform and integrates technologies in **HTML**, **Python**, and **TypeScript** to create interactive, autonomous agent simulacra.

## 🚀 Overview

This repo aims to provide a robust framework for simulating agents driven by generative AI models. These agents can interact, reason, and mimic human traits, supporting research and practical explorations in artificial autonomy, cognitive modeling, and social simulations.

**Key Features:**
- Multi-agent simulation with configurable personalities
- Interactive web-based interfaces (HTML/TypeScript)
- Python backends integrating language models (Ollama LLM, etc.)
- Extensible design for adding behaviors, tasks, and environments

## 📦 Technologies Used

- **HTML** (69.9%): UI construction and visualization
- **Python** (15.6%): Backend logic, data handling, and LLM integration
- **TypeScript** (13.8%): Interactive frontend, dynamic agent behaviors
- **Other** (0.7%): Supporting scripts and files

## 🛠️ Getting Started

Clone the repository:
```sh
git clone https://github.com/fuadmefleh/generative_agents.git
```

### Prerequisites

- Python 3.8+
- Node.js and npm (for TypeScript/HTML frontend)
- Ollama platform and models (see [Ollama documentation](https://ollama.com/))
- Install backend dependencies:
  ```sh
  pip install -r requirements.txt
  ```
- Install frontend dependencies:
  ```sh
  cd frontend
  npm install
  ```

### Running the Application

1. **Start the backend server:**
   ```sh
   cd backend
   uvicorn app.main:app --reload --host 0.0.0.0 --port 9010 
   ```
2. **Start the frontend:**
   ```sh
   cd frontend
   npm run dev
   ```
3. **Configure models as required in `.env` or config files.**

## 📁 Repository Structure

- `/agents` – Core logic for agent behaviors and reasoning (Python)
- `/frontend` – Web UI and interaction layer (HTML/TypeScript)
- `/configs` – Configuration files for agents, models, and environments
- `/utils` – Shared utilities

## 🤖 Example Use Cases

- Social simulations (e.g., emergent group behavior)
- Cognitive modeling and research
- Autonomous conversational agents
- Interactive demos and experimental prototypes

## 📝 Contributing

Contributions, bug reports, and feature suggestions are welcome!
- Open an issue for discussions.
- Make a pull request with clear documentation.

## 📄 License

Distributed under the MIT License. See [LICENSE](LICENSE).

## 👤 Author

Developed by [fuadmefleh](https://github.com/fuadmefleh).

---

For questions or support, please open an issue in this repository.
