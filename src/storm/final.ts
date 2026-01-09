import { HumanMessage } from '@langchain/core/messages';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { ChatGroq } from '@langchain/groq';
import { Annotation, END, MemorySaver, MessagesAnnotation, Send, START, StateGraph } from '@langchain/langgraph';
import dotenv from 'dotenv' ;
import path from 'path' ;
import { z } from 'zod';
import { create_analysts, human_feedback } from './analysts.js';
import { graph_interview } from './conduct_interview.js';
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

const final_state = Annotation.Root({
    /****************************************** analyst state ***************************************/
    topic: Annotation<string>,
    max_analysts: Annotation<number>,
    human_analyst_feedback: Annotation<string>,
    analysts: Annotation<Analyst[]>,
    /*************************************** finalization state *************************************/
    sections: Annotation<string[]>({
        default: () => [],
        reducer: (a, b) => a.concat(b),
    }), /* 
            this state exists in the conduct_interview sub_graph but without reducer ,
            in this final graph we will need that reducer because we will be updating the state in parallel
            without a reducer it will cause errors, we need to tell the state how it should be updated 
        */
    introduction: Annotation<string>,
    content: Annotation<string>,
    conclusion: Annotation<string>,
    final_report: Annotation<string>,
})

function initiate_all_interviews(state: typeof final_state.State) {
    const human_feedback = state.human_analyst_feedback
    if (human_feedback) {
        return 'create_analyst'
    } else {
        return state.analysts.map((analyst)=>
            new Send(
                "conduct_interview",
                {
                    max_num_turns: 2, 
                    analyst ,
                    messages: ["human",`So you said you were writing an article on ${state.topic} ?`] ,
                }
            )
        )
    }
}

const report_writer_instructions = ChatPromptTemplate.fromMessages([
    ["system",`You are a technical writer creating a report on this overall topic: 
    {topic}
    You have a team of analysts. Each analyst has done two things: 
    1. They conducted an interview with an expert on a specific sub-topic.
    2. They write up their finding into a memo.
    Your task: 
    1. You will be given a collection of memos from your analysts.
    2. Think carefully about the insights from each memo.
    3. Consolidate these into a crisp overall summary that ties together the central ideas from all of the memos. 
    4. Summarize the central points in each memo into a cohesive single narrative.
    To format your report:
    1. Use markdown formatting. 
    2. Include no pre-amble for the report.
    3. Use no sub-heading. 
    4. Start your report with a single title header: ## Insights
    5. Do not mention any analyst names in your report.
    6. Preserve any citations in the memos, which will be annotated in brackets, for example [1] or [2].
    7. Create a final, consolidated list of sources and add to a Sources section with the '## Source' header.
    8. List your sources in order and do not repeat.
    [1] Source 1
    [2] Source 2
    Here are the memos from your analysts to build your report from: 
    {context}`],
    ['human',"Write a report based upon these memos."]
])
async function write_report_content(state: typeof final_state.State) {
    const chain = report_writer_instructions.pipe(llm)
    const formatted_sections = state.sections.join('\n\n')
    const report = await chain.invoke({
        topic: state.topic ,
        context : formatted_sections
    })

    return {content: report.content}
}

const intro_conclusion_instructions = ChatPromptTemplate.fromMessages([
    ["system",`You are a technical writer finishing a report on {topic}
    You will be given all of the sections of the report.
    You job is to write a crisp and compelling introduction or conclusion section.
    The user will instruct you whether to write the introduction or conclusion.
    Include no pre-amble for either section.
    Target around 100 words, crisply previewing (for introduction) or recapping (for conclusion) all of the sections of the report.
    Use markdown formatting. 
    For your introduction, create a compelling title and use the # header for the title.
    For your introduction, use ## Introduction as the section header. 
    For your conclusion, use ## Conclusion as the section header.
    Here are the sections to reflect on for writing: {formatted_sections}`],
    ['human',"Write the report {intro_ccl}."]
])
async function write_report_intro(state: typeof final_state.State) {
    const chain = intro_conclusion_instructions.pipe(llm)
    const formatted_sections = state.sections.join('\n\n')
    const introduction = await chain.invoke({
        topic: state.topic , 
        formatted_sections ,
        intro_ccl: 'introduction'
    })

    return {introduction: introduction.content}
}

async function write_report_conclusion(state: typeof final_state.State) {
    const chain = intro_conclusion_instructions.pipe(llm)
    const formatted_sections = state.sections.join('\n\n')
    const conclusion = await chain.invoke({
        topic: state.topic , 
        formatted_sections ,
        intro_ccl: 'conclusion'
    })

    return {conclusion: conclusion.content}
}

async function finalize_report(state: typeof final_state.State) {
    let content = state.content;
    if (content.startsWith("## Insights")) {
        content = content.replace(/^## Insights/, '').trim();
    }
    
    let sources: string | null = null;
    if (content.includes("## Sources")) {
        try {
            const parts = content.split("\n## Sources\n");
            content = parts[0];
            sources = parts[1] || null;
        } catch {
            sources = null;
        }
    }

    let finalReport = `${state.introduction}\n\n---\n\n${content}\n\n---\n\n${state.conclusion}`;
    if (sources !== null) {
        finalReport += `\n\n## Sources\n${sources}`;
    }
    
    return { final_report: finalReport };
}

const final_graph = new StateGraph(final_state)
.addNode('create_analyst',create_analysts)
.addNode('human_feedback',human_feedback)
.addNode('conduct_interview',graph_interview.compile()) // Sub-Graph
.addNode('write_report',write_report_content)
.addNode('write_introduction',write_report_intro)
.addNode('write_conclusion',write_report_conclusion)
.addNode('finalize_report',finalize_report)

.addEdge(START,'create_analyst')
.addEdge('create_analyst','human_feedback')
.addConditionalEdges('human_feedback',initiate_all_interviews,["create_analyst", "conduct_interview"])
.addEdge('conduct_interview','write_report')
.addEdge('conduct_interview','write_introduction')
.addEdge('conduct_interview','write_conclusion')
.addEdge(['write_report','write_introduction','write_conclusion'],'finalize_report')
.addEdge('finalize_report',END)

const memory = new MemorySaver()
export const research_assistant = final_graph.compile({interruptBefore:['human_feedback'],checkpointer: memory})

/* const config = {"configurable": {"thread_id": "15"}}

await research_assistant.invoke({
    max_analysts: 2 ,
    topic: "The benefits of adopting LangGraph as an agent framework"
}, config)

const final_result = await research_assistant.invoke(null , config)

console.log(final_result.final_report) */