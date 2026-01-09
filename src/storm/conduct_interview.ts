import { TavilySearchResults } from '@langchain/community/tools/tavily_search';
import { WikipediaQueryRun } from '@langchain/community/tools/wikipedia_query_run';
import { AIMessage, getBufferString, HumanMessage } from '@langchain/core/messages';
import { ChatPromptTemplate, MessagesPlaceholder } from '@langchain/core/prompts';
import { ChatGroq } from '@langchain/groq';
import { Annotation, END, MessagesAnnotation, START, StateGraph } from '@langchain/langgraph';
import dotenv from 'dotenv' ;
import path from 'path' ;
import { z } from 'zod';
dotenv.config({ path: path.resolve(process.cwd(), '../../.env') });

const llm = new ChatGroq({
  model: "mistral-saba-24b",
  temperature: 0.5,
}) 

interface Analyst {
    affiliation: string;
    name: string;
    role: string;
    description: string;
}

const interview_state = Annotation.Root({
    ...MessagesAnnotation.spec,
    max_num_turns: Annotation<number>,
    context: Annotation<any[]>({
        default: () => [],
        reducer: (a, b) => a.concat(b)
    }),
    analyst: Annotation<Analyst>,
    interview: Annotation<string>,
    sections: Annotation<any[]>
})

const question_instructions = ChatPromptTemplate.fromMessages([
    ["system",`You are an analyst tasked with interviewing an expert to learn about a specific topic. 
    Your goal is boil down to interesting and specific insights related to your topic.
    1. Interesting: Insights that people will find surprising or non-obvious.
    2. Specific: Insights that avoid generalities and include specific examples from the expert.
    Here is your topic of focus and set of goals: {goals}
    Begin by introducing yourself using a name that fits your persona, and then ask your question.
    Continue to ask questions to drill down and refine your understanding of the topic.
    Pay attention and don't ask the same questions.
    When you are satisfied with your understanding, complete the interview with: "Thank you so much for your help!"
    Remember to stay in character throughout your response, reflecting the persona and goals provided to you.`],
    new MessagesPlaceholder('messages')
])

async function generate_question(state: typeof interview_state.State) {
    const chain = question_instructions.pipe(llm)
    const result = await chain.invoke({
        goals: state.analyst,
        messages: state.messages
    })

    return {"messages": [result]}
}

const search_query_schema = z.object({
    search_query: z.string().describe('Search query for retrieval.')
})
const tavily_search = new TavilySearchResults({maxResults: 1})
const search_instructions = ChatPromptTemplate.fromMessages([
    ["system",`You will be given a conversation between an analyst and an expert. 
    Your goal is to generate a well-structured query for use in retrieval and / or web-search related to the conversation.
    First, analyze the full conversation.
    Pay particular attention to the final question posed by the analyst.
    Convert this final question into a well-structured web search query`],
    new MessagesPlaceholder('messages')
])
type TavilyDoc = {
    title: string;
    url: string;
    content: string;
    score: number;
    raw_content: string | null;
};

async function search_web(state: typeof interview_state.State) {
    const structured_llm = llm.withStructuredOutput(search_query_schema)
    const chain = search_instructions.pipe(structured_llm)
    const result = await chain.invoke({
        messages: state.messages
    })
    const response = await tavily_search.invoke({input: result.search_query})
    const parsed = typeof response === "string" ? JSON.parse(response) : response;
    const fromatted_response = parsed.map((doc: TavilyDoc) => {
    return (
        `<Document href="${doc.url}">\n${doc.content}\n</Document>`
    )
    })

    return {context: [fromatted_response]}
}

async function wikipedia(state: typeof interview_state.State) {
    const wiki_search = new WikipediaQueryRun({
        topKResults: 3,
        maxDocContentLength: 1000,
    });
    const structured_llm = llm.withStructuredOutput(search_query_schema)
    const chain = search_instructions.pipe(structured_llm)
    const query_result = await chain.invoke({
        messages: state.messages
    })
    const docs = await wiki_search.invoke(query_result.search_query)
    
    return {context: [docs]}
}

const answer_instructions = ChatPromptTemplate.fromMessages([
    ["system",`You are an expert being interviewed by an analyst.
    Here is analyst area of focus: {goals}. 
    You goal is to answer a question posed by the interviewer.
    To answer question, use this context:
    {context}
    When answering questions, follow these guidelines:
    1. Use only the information provided in the context. 
    2. Do not introduce external information or make assumptions beyond what is explicitly stated in the context.
    3. The context contain sources at the topic of each individual document.
    4. Include these sources your answer next to any relevant statements. For example, for source # 1 use [1]. 
    5. List your sources in order at the bottom of your answer. [1] Source 1, [2] Source 2, etc
    6. If the source is: <Document source="assistant/docs/llama3_1.pdf" page="7"/>' then just list: 
    [1] assistant/docs/llama3_1.pdf, page 7 
    And skip the addition of the brackets as well as the Document source preamble in your citation.`],
    new MessagesPlaceholder('messages')
])

async function generate_answer(state: typeof interview_state.State) {
    const chain = answer_instructions.pipe(llm)
    const result = await chain.invoke({
        goals: state.analyst ,
        context: state.context ,
        messages: state.messages ,
    })
    result.name = "expert"
    console.log('this is the answear after the retriever: ',result)

    return { messages: [result] }
}

function save_interview(state: typeof interview_state.State) {
    const interview_transcript = getBufferString(state.messages)

    return {interview: interview_transcript}
}

function route_message(state: typeof interview_state.State) {
    const messages_length = state.messages.filter(msg => 
        msg instanceof AIMessage && 
        msg.name === 'expert'
    )
    console.log(messages_length)
    console.log('this is the length of the conversation: ',messages_length.length)
    if (2 >= state.max_num_turns) {
        return 'save_interview'
    }
    /* const analyst_messages = state.messages.filter(msg => 
        msg instanceof AIMessage && 
        msg.name !== 'expert'
    )
    const lastMessage = analyst_messages[analyst_messages.length - 1]?.content;
    if (lastMessage?.toString().toLowerCase().includes('Thank you so much for your help')) {
        return 'save_interview'
    }
 */
    return 'ask_question'
}

const section_writer_instructions = ChatPromptTemplate.fromMessages([
    ["system",`You are an expert technical writer. 
            
    Your task is to create a short, easily digestible section of a report based on a set of source documents.

    1. Analyze the content of the source documents: 
    - The name of each source document is at the start of the document, with the <Document tag.
            
    2. Create a report structure using markdown formatting:
    - Use ## for the section title
    - Use ### for sub-section headers
            
    3. Write the report following this structure:
    a. Title (## header)
    b. Summary (### header)
    c. Sources (### header)

    4. Make your title engaging based upon the focus area of the analyst: 
    {focus}

    5. For the summary section:
    - Set up summary with general background / context related to the focus area of the analyst
    - Emphasize what is novel, interesting, or surprising about insights gathered from the interview
    - Create a numbered list of source documents, as you use them
    - Do not mention the names of interviewers or experts
    - Aim for approximately 400 words maximum
    - Use numbered sources in your report (e.g., [1], [2]) based on information from source documents
            
    6. In the Sources section:
    - Include all sources used in your report
    - Provide full links to relevant websites or specific document paths
    - Separate each source by a newline. Use two spaces at the end of each line to create a newline in Markdown.
    - It will look like:

    ### Sources
    [1] Link or Document name
    [2] Link or Document name

    7. Be sure to combine sources. For example this is not correct:

    [3] https://ai.meta.com/blog/meta-llama-3-1/
    [4] https://ai.meta.com/blog/meta-llama-3-1/

    There should be no redundant sources. It should simply be:

    [3] https://ai.meta.com/blog/meta-llama-3-1/
            
    8. Final review:
    - Ensure the report follows the required structure
    - Include no preamble before the title of the report
    - Check that all guidelines have been followed`],
    ["human","Use this source to write your section: {context}"]
])

async function write_section(state: typeof interview_state.State) {
    const chain = section_writer_instructions.pipe(llm)
    const result = await chain.invoke({
        focus: state.analyst.description,
        context: state.context
    })

    return {"sections": [result.content]}
}

export const graph_interview = new StateGraph(interview_state)
.addNode("ask_question", generate_question)
.addNode("search_web", search_web)
.addNode("search_wikipedia", wikipedia)
.addNode("answer_question", generate_answer)
.addNode("save_interview", save_interview)
.addNode("write_section", write_section)
.addEdge(START,"ask_question")
.addEdge("ask_question","search_wikipedia")
.addEdge("ask_question","search_web")
.addEdge(["search_wikipedia","search_web"],"answer_question")
.addConditionalEdges("answer_question",route_message,["ask_question","save_interview"])
.addEdge("save_interview","write_section")
.addEdge("write_section",END)

/* const app = graph_interview.compile().withConfig({runName: "Conduct Interviews"})
const analyst = {
    affiliation:'Tech Innovators Inc.',
    name:'Dr. Emily Carter',
    role:'Technology Analyst',
    description:'Dr. Carter focuses on evaluating emerging technologies and their potential impact on various industries. She is particularly interested in how LangGraph can streamline processes and improve efficiency in tech-driven companies'
}
const messages = [new HumanMessage("So you said you were writing an article on The benefits of adopting LangGraph as an agent framework ?")]
const interview = await app.invoke({
    analyst ,
    messages ,
    max_num_turns: 2
})
console.log(interview) */