/* Pure, bounded graph operations shared by explorers; no dataset or model assumptions. */
(function(global){
  'use strict';
  function mergeGraph(current, incoming, limits){
    const nodes=new Map(current.nodes.map(node=>[node.id,node]));
    for(const node of incoming.nodes)if(nodes.has(node.id)||nodes.size<limits.nodes)nodes.set(node.id,node);
    const edges=new Map(current.edges.map(edge=>[edge.id,edge]));
    for(const edge of incoming.edges){
      if(nodes.has(edge.source)&&nodes.has(edge.target)&&(edges.has(edge.id)||edges.size<limits.edges))edges.set(edge.id,edge);
    }
    return {nodes:[...nodes.values()],edges:[...edges.values()]};
  }
  function structure(graph){
    const neighbors=new Map(graph.nodes.map(node=>[node.id,new Set()]));
    let selfLoops=0;
    for(const edge of graph.edges){
      neighbors.get(edge.source)?.add(edge.target);neighbors.get(edge.target)?.add(edge.source);
      if(edge.source===edge.target)selfLoops++;
    }
    let components=0,isolated=0;const visited=new Set();
    for(const node of graph.nodes){
      if(!neighbors.get(node.id).size)isolated++;
      if(visited.has(node.id))continue;
      components++;const queue=[node.id];visited.add(node.id);
      for(let index=0;index<queue.length;index++)for(const id of neighbors.get(queue[index])||[])if(!visited.has(id)){visited.add(id);queue.push(id);}
    }
    return {components,isolated,selfLoops};
  }
  global.DatasetGraphModel={mergeGraph,structure};
  if(typeof module!=='undefined')module.exports=global.DatasetGraphModel;
})(globalThis);
