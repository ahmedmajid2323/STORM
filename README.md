# STORM: Deep Research Assistant

**Synthesis of Topic Outlines through Retrieval and Multi-perspective Question Asking**

A sophisticated research assistant built with LangGraph that generates comprehensive, Wikipedia-style articles on any user-provided topic by leveraging multiple AI perspectives and parallel research capabilities.

## 🎯 Project Overview

STORM is a deep research assistant designed to explore and demonstrate core LangGraph concepts through a practical, real-world application. The system generates high-quality research articles by:

1. **Creating a team of AI analysts** - Each analyst focuses on a different sub-topic or perspective
2. **Human-In-The-Loop refinement** - Users can refine and adjust sub-topics before research begins
3. **Parallel expert interviews** - Each analyst interviews expert AIs using multiple data sources (Web Search + Wikipedia)
4. **Map/Reduce synthesis** - Expert responses are gathered in parallel and then synthesized into coherent sections
5. **Final article generation** - All perspectives are combined into a comprehensive, well-structured article

## 🧠 Project Goals

This project serves as an exploration of advanced LangGraph capabilities:

- **Human-In-The-Loop** - Interactive refinement of research direction
- **Map/Reduce** - Parallel task execution and result aggregation
- **SubGraphs** - Modular, reusable graph components
- **Prompt Engineering** - Optimized prompts for research and synthesis
- **Memory & Controllability** - Full control over state management and agent behavior

## 🛠️ Tools & Technologies

### Core Framework
- **LangGraph** - State management and agent orchestration
- **LangChain** - LLM integration and tooling
- **TypeScript** - Type-safe development

### AI Models & Services
- **Groq** - High-performance LLM inference (Mistral models)
- **Tavily Search** - Web search capabilities
- **Wikipedia API** - Structured knowledge retrieval
- **OpenWeatherMap API** - Weather data (for example agents)

### Libraries
- `@langchain/langgraph` - Graph-based agent framework
- `@langchain/groq` - Groq LLM integration
- `@langchain/community` - Community tools (Tavily, Wikipedia)
- `dotenv` - Environment variable management
- `zod` - Schema validation
- `cheerio` - HTML parsing
- `chromadb` - Vector database (for potential RAG extensions)

## 📋 Prerequisites

Before you begin, ensure you have the following installed:

- **Node.js** (version 20 or higher)
- **npm** or **yarn** package manager
- **Groq API Key** - Get one from [Groq Console](https://console.groq.com/)
- **Tavily API Key** (optional, for web search)
- **OpenWeatherMap API Key** (optional, for weather tool)

## 🚀 Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd LangGraph
```

### 2. Install Dependencies

```bash
npm install
```

### 3. Configure Environment Variables

Create a `.env` file in the project root directory:

```env
# Required: Groq API Key
# Get your API key from https://console.groq.com/
# Make sure to select a valid and working model from the Groq console
GROQ_API_KEY=your_groq_api_key_here

# Optional: Tavily API Key (for web search)
TAVILY_API_KEY=your_tavily_api_key_here

# Optional: OpenWeatherMap API Key (for weather tool)
OPENWEATHER_API_KEY=your_openweather_api_key_here
```

**Important:** 
- You must select a **valid and working model** from the Groq console that matches your API key
- The default model used is `mistral-saba-24b`, but you can modify this in the source files if needed
- Ensure your Groq account has access to the model you're using

### 4. Build the Project

```bash
npm run build
```

This compiles TypeScript files to JavaScript in the `build/` directory.

## 🏃 Running the Project

### Start the LangGraph Development Server

To start the project, run:

```bash
npx @langchain/langgraph-cli@latest dev
```

This command:
- Starts the LangGraph development server
- Loads the graphs defined in `langgraph.json`
- Makes the agents available via the LangGraph API
- Enables hot-reloading during development

### Available Agents

The project includes two main agents configured in `langgraph.json`:

1. **`storm_agent`** - The main STORM research assistant (`./src/storm/final.ts:research_assistant`)
2. **`simple_agent`** - A simple example agent with weather tool (`./src/agent.ts:app`)

### Development Mode

For local development with nodemon (auto-reload):

```bash
npm run dev
```

## 📁 Project Structure

```
LangGraph/
├── src/
│   ├── agent.ts              # Simple agent example with weather tool
│   ├── htl.ts                # Human-In-The-Loop example
│   ├── map_reduce.ts         # Map/Reduce pattern example
│   ├── parallel/
│   │   ├── research.ts       # Parallel research example
│   │   └── parallelization.ts
│   └── storm/                # STORM research assistant
│       ├── analysts.ts       # Analyst creation logic
│       ├── conduct_interview.ts  # Interview subgraph
│       └── final.ts          # Main STORM graph
├── build/                    # Compiled JavaScript (gitignored)
├── langgraph.json            # LangGraph configuration
├── package.json              # Dependencies and scripts
├── tsconfig.json             # TypeScript configuration
└── .env                      # Environment variables (gitignored)
```

## 🔄 How STORM Works

### Workflow Overview

1. **Topic Input** - User provides a research topic
2. **Analyst Generation** - System creates multiple AI analysts, each with a unique perspective
3. **Human Feedback** - Optional refinement of analysts and sub-topics
4. **Parallel Interviews** - Each analyst interviews expert AIs using:
   - Web search (Tavily)
   - Wikipedia queries
   - (Extensible to other sources like WebBaseLoader, RAG indexes, etc.)
5. **Section Generation** - Each analyst produces a section based on their research
6. **Article Synthesis** - All sections are combined into a coherent article with:
   - Introduction
   - Content sections
   - Conclusion
7. **Final Report** - Complete, well-structured research article

### Key Features

- **Multi-perspective research** - Different analysts explore different angles
- **Parallel processing** - Multiple interviews happen simultaneously (Map pattern)
- **Result aggregation** - All findings synthesized into one document (Reduce pattern)
- **Human oversight** - Users can guide and refine the research process
- **Extensible sources** - Easy to add new data sources (RAG, custom APIs, etc.)

## 🔧 Configuration

### Model Selection

The project uses Groq's Mistral models by default. To change the model, update the model name in the source files:

```typescript
const llm = new ChatGroq({
  model: "mistral-saba-24b",  // Change this to your preferred model
  temperature: 0.5,
})
```

Available models can be found in the [Groq Console](https://console.groq.com/).

### Graph Configuration

The `langgraph.json` file defines which graphs are available:

```json
{
  "node_version": "20",
  "dependencies": ["."],
  "graphs": {
    "storm_agent": "./src/storm/final.ts:research_assistant",
    "simple_agent": "./src/agent.ts:app"
  },
  "env": "./.env"
}
```

## 📝 Example Usage

Once the development server is running, you can interact with the STORM agent through the LangGraph API or by modifying the source files to add test invocations.

## 🔒 Security Notes

- Never commit your `.env` file to version control
- The `.gitignore` file is configured to exclude:
  - `.env` files
  - `node_modules/`
  - `build/` directory
- All API keys should be stored in environment variables, never hardcoded

## 🤝 Contributing

This is a learning project exploring LangGraph capabilities. Feel free to:
- Experiment with different models
- Add new data sources
- Implement additional research strategies
- Improve prompt engineering

## 📚 Learn More

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Documentation](https://js.langchain.com/)
- [Groq Documentation](https://console.groq.com/docs)
- [STORM Paper](https://arxiv.org/abs/2402.14207) - Original research paper that inspired this project

## 📄 License

ISC

---

**Note:** This project is a side project for exploring LangGraph concepts. It's designed to be educational and experimental rather than production-ready.

