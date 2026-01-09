import { TavilySearchResults } from "@langchain/community/tools/tavily_search";
import { HumanMessage, AIMessage } from "@langchain/core/messages";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { StateGraph, MessagesAnnotation, MemorySaver } from "@langchain/langgraph";
import { ChatGroq } from "@langchain/groq";
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

function convertUnixTimestampToTime(timestamp : number, timezoneOffset: number) : string {
  // Convert to milliseconds
  const date = new Date((timestamp + timezoneOffset) * 1000);
  
  // Extract hours and minutes
  let hours = date.getUTCHours();
  let minutes = date.getUTCMinutes();
  
  // Format hours in 12-hour format
  const ampm = hours >= 12 ? 'PM' : 'AM';
  hours = hours % 12 || 12; // Convert 0 to 12 for AM/PM format
  
  // Ensure two-digit minutes
  const minutesStr = minutes.toString().padStart(2, '0'); // Convert minutes to string properly

  return `${hours}:${minutesStr} ${ampm}`;
}

const prompt_weather = ChatPromptTemplate.fromTemplate(`
  Given the following weather data: {context}, 
  generate a friendly and engaging weather description. 
  Avoid listing raw data points. Instead, summarize the weather naturally, highlighting key details like temperature, cloud cover, wind, and any significant conditions. 
  Convert the sunrise and sunset times from Unix timestamps to a human-readable format (HH:MM AM/PM).
  Make it feel like a weather report someone would read on a weather app.
`)
const weather_chain = prompt_weather.pipe(llm)

const get_weather_tool = tool(
  async (zone: string) => {

    if (!zone) {
      return "❌ Error: No location provided.";
    }

    const param = zone.toLowerCase()
    const apiKey = process.env.OPENWEATHER_API_KEY;
    if (!apiKey) {
      return "❌ Error: OpenWeather API key is not configured. Please set OPENWEATHER_API_KEY in your .env file.";
    }
    const response = await fetch(`https://api.openweathermap.org/data/2.5/weather?q=${param}&units=metric&appid=${apiKey}`);
    const data = await response.json();

    if (data.cod === "404") { 
      return `❌ Error: ${data.message}`
    } else {

      const weather_data = {
        sunrise: convertUnixTimestampToTime(data.sys.sunrise , data.timezone) ,
        sunset: convertUnixTimestampToTime(data.sys.sunset , data.timezone) ,
        temp: data.main.temp ,
        clouds: data.clouds.all ,
      }
      const result = await weather_chain.invoke({context: weather_data})
  
      return result.content;
    }

  },
  {
    name: 'weather_tool',
    description:'this tool is useful if the user wants to have access to the weather in a specific country or region',
    schema: z.string().describe(
      "The name of the region or country for which the weather information is requested." +
      "Provide only the name of the location (e.g., Tunis, Paris, Germany) without any additional text or formatting."
    ),
    returnDirect: true
  }
)

const tools = [get_weather_tool];
//the TavilySearchResults tool comes with built-in metadata (name, description, and schema) 
// because it's a pre-defined tool class that extends LangChain's StructuredTool or Tool base class.
const toolNode = new ToolNode(tools);

const model = llm.bindTools(tools);

//defining nodes
function shouldContinue({ messages }: typeof MessagesAnnotation.State) {
  const lastMessage = messages[messages.length - 1] as AIMessage;

  // If the LLM makes a tool call
  if (lastMessage.tool_calls?.length) {
    return "tools"; // routing to the "tools" node
  }
  // Otherwise, we stop (reply to the user) using the special "__end__" node
  return "__end__";
}

async function callModel(state: typeof MessagesAnnotation.State) {
  const response = await model.invoke(state.messages);

  // We return a list, because this will get added to the existing list
  return { messages: [response] };
}

// Define a new graph
const workflow = new StateGraph(MessagesAnnotation)
  .addNode("agent", callModel)
  .addEdge("__start__", "agent") // __start__ is a special name for the entrypoint
  .addNode("tools", toolNode)
  .addEdge("tools", "agent")
  .addConditionalEdges("agent", shouldContinue); // check the toolsCondition as a built-in conditional edge

// Finally, we compile it into a LangChain Runnable.
const memory = new MemorySaver() // define checkpointer
export const app = workflow.compile(/* {checkpointer: memory} */);

const config = {"configurable": {"thread_id":"1"}}
const finalState = await app.invoke({
  messages: [
    new HumanMessage("hi , my name is ahmed"),
  ],
}, config);
console.log(finalState.messages[finalState.messages.length - 1].content);

/* const nextState = await app.invoke({
  // Including the messages from the previous run gives the LLM context.
  messages: [...finalState.messages, new HumanMessage("what about ny")],
});
console.log(nextState.messages[nextState.messages.length - 1].content); */