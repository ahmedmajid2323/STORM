import { ChatGroq } from "@langchain/groq";
import { Annotation, END, Send, START, StateGraph } from "@langchain/langgraph";
import dotenv from 'dotenv' ;
import path from 'path' ;
import { tool } from "@langchain/core/tools";
import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";
import { z } from "zod";
dotenv.config({ path: path.resolve(process.cwd(), '../.env') });

const llm = new ChatGroq({
  model: "mistral-saba-24b",
  temperature: 0.5,
}) 

//(1) Map - Break a task into smaller sub-tasks, processing each sub-task in parallel.
//(2) Reduce - Aggregate the results across all of the completed, parallelized sub-tasks.

//Let's design a system that will do two things:

//(1) Map - Create a set of jokes about a topic, in parallel ,
//(2) Reduce - Pick the best joke from the list.

const subjects_prompt = "Generate a comma separated list of between 2 and 5 examples related to: {topic}.."
const joke_prompt = "Generate a joke about {subject}"
const best_joke_prompt = ChatPromptTemplate.fromTemplate(
    "Below are a bunch of jokes about {topic}. Select the best one! Return the ID of the best one, starting 0 as the ID for the first joke. Jokes: \n\n  {jokes}"
)

const state_graph = Annotation.Root({
    topic: Annotation<string>,
    subjects: Annotation<string[]>,
    jokes: Annotation<string[]>({
        default: () => [],
        reducer: (a, b) => a.concat(b),
    }),
    best_selected_joke: Annotation<string>
})

const subjects_schema =  z.object({
    subjects: z.array(z.string()),
  });
async function generate_subjects(state: typeof state_graph.State) {
    const prompt = subjects_prompt.replace("topic" , state.topic)
    const response = await llm.withStructuredOutput(subjects_schema).invoke(prompt) // use withStructuredOutput with an object zod !!!
    return { subjects : response.subjects }
}

// *** "Send" allow you to pass any state that you want to generate_joke! It does not have to align with OverallState.
// Send takes two arguments: first is the name of the node, and second is the state to pass to that node.
const continueToJokes = (state: typeof state_graph.State) => {
    return state.subjects.map((subject) => new Send("generateJoke", { subject }));
    // the " generateJoke " node could have any state we want
};

interface JokeState {
    subject: string;
}
const joke_schema = z.object({
    joke: z.string(),
  });
async function generateJoke(state: JokeState): Promise<{jokes : string[]}> {// this is the node that we applied **Send** to ot
    const prompt = joke_prompt.replace('subject' , state.subject)
    const response = await llm
    .withStructuredOutput(joke_schema , {name: 'joke'}) // {name: 'joke'} just a metadata (label), c rien
    .invoke(prompt)

    return { jokes: [response.joke] };
}

const BestJoke = z.object({
    id: z.number(),
});
async function best_joke(state: typeof state_graph.State) {
    const jokes_array = state.jokes.join('\n\n')
    const structured_llm = llm.withStructuredOutput(BestJoke)
    const chain = best_joke_prompt.pipe(structured_llm)
    const response = await chain.invoke({
        topic: state.topic ,
        jokes: jokes_array
    })

    return { best_selected_joke: state.jokes[response.id] }
}

const graph = new StateGraph(state_graph)
.addNode('generate_subjects' , generate_subjects)
/* .addNode('continueToJokes' , continueToJokes) => c pas un node */
.addNode('generateJoke' , generateJoke)
.addNode('best_joke' , best_joke)
.addEdge(START , 'generate_subjects')
.addConditionalEdges('generate_subjects' , continueToJokes ) // continueToJokes will **Send to generate the jokes
.addEdge('generateJoke' , 'best_joke')
.addEdge('best_joke' , END)

const app = graph.compile() 

const res = await app.invoke({topic : 'animals'})
console.log(res)