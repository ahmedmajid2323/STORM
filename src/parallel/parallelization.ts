import { Annotation, END, START, StateGraph } from "@langchain/langgraph";

const state_graph = Annotation.Root({
    state: Annotation<string[]>({
        default: () => [],
        reducer: (a, b) => a.concat(b),
      }),
})

function node_A(state: typeof state_graph.State) {
    console.log(`adding node A to {state : ${state.state}}`)
    return { state : ['node_A'] }
}

function node_B(state: typeof state_graph.State) {
    console.log(`adding node B to {state : ${state.state}}`)
    return { state : ['node_B'] }
}

function node_B2(state: typeof state_graph.State) { // in this case we ran (B => B2) & C in parallel
    console.log(`adding node B2 to {state : ${state.state}}`)
    return { state : ['node_B2'] }
}

function node_C(state: typeof state_graph.State) {
    console.log(`adding node C to {state : ${state.state}}`)
    return { state : ['node_C'] }
}

function node_D(state: typeof state_graph.State) {
    console.log(`adding node D to {state : ${state.state}}`)
    return { state : ['node_D'] }
}

const graph = new StateGraph(state_graph)
.addNode('node_A',node_A)
.addNode('node_B',node_B)
.addNode('node_B2',node_B2)
.addNode('node_C',node_C)
.addNode('node_D',node_D)
.addEdge(START, "node_A")
.addEdge("node_A", "node_B")
.addEdge("node_A", "node_C")
.addEdge("node_B", "node_B2")
.addEdge(["node_C" , "node_B2"], "node_D") // node_D waits till the parallel nodes finish their tasks
.addEdge("node_D", END)
// in this example the order of execution was B => C => B2 , but you can adjust the order ==> Stable Sorting

const app = graph.compile()

const res = await app.invoke({state: []})

console.log(res)