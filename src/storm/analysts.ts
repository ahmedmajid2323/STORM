import { ChatPromptTemplate } from '@langchain/core/prompts';
import { ChatGroq } from '@langchain/groq';
import { Annotation, END, MemorySaver, START, StateGraph } from '@langchain/langgraph';
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

const analyst_state = Annotation.Root({
    topic: Annotation<string>,
    max_analysts: Annotation<number>,
    human_analyst_feedback: Annotation<string>,
    analysts: Annotation<Analyst[]>
})

const analyst_intructions = ChatPromptTemplate.fromMessages([
    ["system", `You are tasked with creating a set of AI analyst personas. Follow these instructions carefully:
        1. First, review the research topic: {topic}
        2. Examine any editorial feedback that has been optionally provided to guide creation of the analysts: {human_analyst_feedback}
        3. Determine the most interesting themes based upon documents and / or feedback above.
        4. Pick the top {max_analysts} themes.
        5. Assign one analyst to each theme.`],
    ["human", "Generate the set of analysts"]
]);

const analystSchema = z.object({
    affiliation : z.string().describe("Primary affiliation of the analyst.") , 
    name : z.string().describe("Name of the analyst.") , 
    role : z.string().describe("Role of the analyst in the context of the topic.") , 
    description : z.string().describe("Description of the analyst focus, concerns, and motives.") , 
})

const perspectives =z.object({
    analysts: z.array(analystSchema)
    .describe('Comprehensive list of analysts with their roles and affiliations.')
}) 

export async function create_analysts(state: typeof analyst_state.State) : Promise<{analysts: Analyst[]}> {

    // the schema given to withStructuredOutput must be an object !!
    const structured_llm = llm.withStructuredOutput(perspectives , { name: "Perspectives" }); 
    const chain = analyst_intructions.pipe(structured_llm)

    const response = await chain.invoke({
        topic: state.topic,
        human_analyst_feedback: state.human_analyst_feedback,
        max_analysts: state.max_analysts
    });

    console.log('Analysts created: ', response.analysts )
    
    return {analysts: response.analysts}
}

export function human_feedback(state: typeof analyst_state.State) {
    // No-op - just return the state unchanged
    return state
}

/* function should_continue(state: typeof analyst_state.State) {
    const human_feedback = state.human_analyst_feedback

    if (human_feedback) {
        return 'create_analysts'
    } else {
        return END 
    }
}

const graph = new StateGraph(analyst_state)
.addNode('create_analysts' , create_analysts)
.addNode('human_feedback' , human_feedback) 
.addEdge(START , 'create_analysts')
.addEdge('create_analysts' , 'human_feedback')
.addConditionalEdges('human_feedback' , should_continue, ["create_analysts" , END])

const memory = new MemorySaver()
const app = graph.compile({checkpointer: memory , interruptBefore: ['human_feedback']})

const thread = {"configurable": {"thread_id": "1"}}
const response = await app.invoke({
    max_analysts: 3 ,
    topic: "The benefits of adopting LangGraph as an agent framework"
}, thread)

console.log('this is the response of the agent', response)

console.log('-------------------------- after updating the state --------------------------')

await app.updateState(thread , {human_analyst_feedback : "Add in someone from a startup to add an entrepreneur perspective"})

await app.invoke(null , thread)

await app.updateState(thread , {human_analyst_feedback : ""})

const final_response = await app.invoke(null , thread)

console.log(final_response) */
