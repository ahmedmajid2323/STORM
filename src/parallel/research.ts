import { TavilySearchResults } from "@langchain/community/tools/tavily_search";
import { WikipediaQueryRun } from "@langchain/community/tools/wikipedia_query_run";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { ChatGroq } from "@langchain/groq";
import { Annotation, END, START, StateGraph } from "@langchain/langgraph";
import dotenv from 'dotenv' ;
import path from 'path' ;
dotenv.config({ path: path.resolve(process.cwd(), '../../.env') });

const llm = new ChatGroq({
  model: "mistral-saba-24b",
  temperature: 0.5,
}) 

const state_graph = Annotation.Root({
    question: Annotation<string>,
    search_query: Annotation<string>,
    answear: Annotation<string>,
    context: Annotation<string[]>({
        default: () => [],
        reducer: (a, b) => a.concat(b),
    }),
})

async function llm_node(state: typeof state_graph.State) {
    const prompt = ChatPromptTemplate.fromTemplate(
        `You are a helpful assistant that extracts concise search topics.
        Given a natural language question or request, extract a short and specific topic or keyword that best describes the core subject. 
        The topic should be suitable for a web search — avoid full sentences, and keep it brief and to the point.

        ### Examples:
        Question: "What are the long-term effects of climate change on coastal cities?"
        Topic: climate change coastal effects

        Question: "How does GPT-4 compare to GPT-3.5 in terms of reasoning?"
        Topic: GPT-4 vs GPT-3.5 reasoning

        Question: "Can drinking green tea improve focus during study sessions?"
        Topic: green tea focus benefits
        
        now given this question : {question}, \n give me the tpic :`
    )
    const chain = prompt.pipe(llm)
    const result = await chain.invoke({question: state.question})
    return { search_query : result.content}
}

type TavilyDoc = {
    title: string;
    url: string;
    content: string;
    score: number;
    raw_content: string | null;
};

async function tavily_node(state: typeof state_graph.State) {
    const tavily_search = new TavilySearchResults({maxResults: 3})
    const response = await tavily_search.invoke({input: state.search_query}) 
    const parsed = typeof response === "string" ? JSON.parse(response) : response;
    // returns smth like : [{"title":"..","url":"https://..","content":"..","score":0.94,"raw_content":null} , ...]
    const fromatted_response = parsed.map((doc: TavilyDoc) => {
        return (
            `<Document href="${doc.url}">\n${doc.content}\n</Document>`
        )
    })
    return { context : fromatted_response }
}

async function wiki_node(state: typeof state_graph.State) {
    const wiki_search = new WikipediaQueryRun({
        topKResults: 3,
        maxDocContentLength: 4000,
    });
    const response = await wiki_search.invoke(state.search_query)
    return { context : response }
}

async function generate_answear(state: typeof state_graph.State) {
    const prompt = ChatPromptTemplate.fromTemplate(
        'Answer the question {question} using this context: {context}'
    )
    const chain = prompt.pipe(llm)
    const response = await chain.invoke({
        question : state.question ,
        context : state.context ,
    })
    return { answear: response.content }
}

const graph = new StateGraph(state_graph)
.addNode('formulate_query' , llm_node )
.addNode('tavily_search' , tavily_node )
.addNode('wiki_search' , wiki_node )
.addNode('gen_answear' , generate_answear )
.addEdge(START , 'formulate_query' )
.addEdge('formulate_query' , 'tavily_search' )
.addEdge('formulate_query' , 'wiki_search' )
.addEdge(['tavily_search' , 'wiki_search']  , 'gen_answear')
.addEdge('gen_answear' , END)

const app = graph.compile()

const response = await app.invoke({question: "How were Nvidia's Q2 2024 earnings"})
console.log(response.answear)